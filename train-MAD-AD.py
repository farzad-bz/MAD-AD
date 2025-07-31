
import torch
# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
from collections import OrderedDict
import json
from copy import deepcopy
from glob import glob
from time import time
import argparse
import logging
import os
import random
from models import UNET_models

from diffusion_x0 import create_diffusion
from MedicalDataLoader import MedicalDataset
from transformers import get_cosine_schedule_with_warmup

from huggingface_hub import hf_hub_download


#################################################################################
#                             Training Helper Functions                         #
#################################################################################



def cleanup():
    """
    End DDP training.
    """
    dist.destroy_process_group()


def set_random_seed():
    """
    Sets a random seed for Python's random module, NumPy, and PyTorch.
    This ensures randomness in subsequent operations.
    """
    # Generate a random seed
    seed = random.randint(0, 2**32 - 1)

    # Set the seed for all libraries
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # If using multi-GPU

    # Disable deterministic behavior for cuDNN to allow randomness
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # If using multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    if dist.get_rank() == 0:  # real logger
        logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
        )
        logger = logging.getLogger(__name__)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger

def random_mask(x : torch.Tensor, mask_ratios, mask_patch_size_x=1, mask_patch_size_y=1):
    for mask_ratio in mask_ratios:
        assert mask_ratio >=0 and mask_ratio<=1
    n, c, w, h = x.shape
    size = int(np.prod(x.shape[2:]) / (mask_patch_size_x*mask_patch_size_y))
    mask = torch.zeros((n,c,size)).to(x.device)
    for b in range(n):
        masked_indexes = np.arange(size)
        np.random.shuffle(masked_indexes)
        masked_indexes = masked_indexes[:int(size * (1 - mask_ratios[b]))]
        mask[b,:, masked_indexes] = 1
    mask = mask.reshape(n, c, int(w/mask_patch_size_x), int(w/mask_patch_size_y))
    mask = mask.repeat_interleave(mask_patch_size_x, dim=2).repeat_interleave(mask_patch_size_y, dim=3)
    return mask


def shuffle_patches(image, patch_size_x, patch_size_y):
    N, C, H, W = image.shape
    P_x = patch_size_x
    P_y = patch_size_y
    assert H % P_y == 0 and W % P_x == 0, "Image dimensions should be divisible by patch size."

    # Extract patches
    unfolded = torch.nn.functional.unfold(image, kernel_size=(patch_size_x, patch_size_y), stride=(patch_size_x, patch_size_y))  # Shape: (N*C*P*P, num_patches)

    # Reshape unfolded patches to (N, C, P, P, num_patches)
    num_patches = unfolded.shape[-1]
    unfolded = unfolded.view(N, C, P_x, P_y, num_patches)

    # Shuffle patches across the batch dimension
    unfolded = unfolded.permute(0, 4, 1, 2, 3)  # Shape: (N, num_patches, C, P, P)
    unfolded = unfolded.reshape(N * num_patches, C, P_x, P_y)  # Shape: (N * num_patches, C, P, P)

    # Shuffle patches
    indices = torch.randperm(N * num_patches)
    shuffled_unfolded = unfolded[indices]

    # Reshape back to original format
    shuffled_unfolded = shuffled_unfolded.view(N, num_patches, C, P_x, P_y)
    shuffled_unfolded = shuffled_unfolded.permute(0, 2, 3, 4, 1)  # Shape: (N, C, P, P, num_patches)

    # Reconstruct the image
    shuffled_unfolded = shuffled_unfolded.contiguous().view(N * C * P_x * P_y, num_patches)
    folded = torch.nn.functional.fold(shuffled_unfolded, output_size=(H, W), kernel_size=(patch_size_x, patch_size_y), stride=(patch_size_x, patch_size_y))

    # Fold operation does not include channels; need to reshape and combine
    folded = folded.view(N, C, H, W)
    
    return folded


#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):
    """
    Trains a new model.
    """
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    # Setup DDP:
    dist.init_process_group("nccl")
    assert args.global_batch_size % dist.get_world_size() == 0, f"Batch size must be divisible by world size."
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    # Setup an experiment folder:
    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        with open(f'{args.results_dir}/args.txt', 'w') as f:
            json.dump(args.__dict__, f, indent=2)
        experiment_index = len(glob(f"{args.results_dir}/*"))
        experiment_dir = f"{args.results_dir}/{experiment_index:03d}-{args.model_size}"  # Create an experiment folder
        checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")
    else:
        logger = create_logger(None)

    # Create model:
    assert args.image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = args.image_size // 8
    model = UNET_models[args.model_size](latent_size=latent_size)
    model = DDP(model.to(device), device_ids=[rank])
    diffusion = create_diffusion(timestep_respacing="ddim10", predict_xstart=True, sigma_small=False, learn_sigma = args.learn_sigma, diffusion_steps=10)  # default: 1000 steps, linear noise schedule

    vae_model_path = hf_hub_download(repo_id="farzadbz/Medical-VAE", filename="VAE-Medical-klf8.pt")
    
    # Load the model
    vae = torch.load(vae_model_path)
    vae.eval()

    
    vae.to(device)
    logger.info(f"Number of Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Setup data:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5], inplace=True)
    ])
    
    dataset = MedicalDataset('train', rootdir=args.train_data_root, modality=args.modaity, transform=transform, image_size=args.image_size, augment=args.augmentation, modality=args.modality)
    loader = DataLoader(dataset, batch_size=args.global_batch_size, shuffle=True, num_workers=4, drop_last=False)
    
    
    val_dataset = MedicalDataset('val', rootdir=args.val_data_root, modality=args.modaity, transform=transform, image_size=args.image_size, augment=False, modality=args.modality)
    val_loader = DataLoader(val_dataset, batch_size=args.global_batch_size, shuffle=False, num_workers=4, drop_last=False)
    
    accumulation_steps = 1
        
    logger.info(f"Dataset contains {len(dataset):,} training images and  {len(val_dataset):,} validation images")

    adjusted_epochs = args.epochs

    # Setup optimizer (we used default Adam betas=(0.9, 0.999) and a constant learning rate of 1e-4 in our paper):
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = get_cosine_schedule_with_warmup(
        opt,
        num_warmup_steps=args.warmup_epochs,
        num_training_steps=args.epochs*1.2,
    )
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=adjusted_epochs, eta_min=args.lr/100)
    # Setup data:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5], inplace=True)
    ])
        

    # Prepare models for training
    model.train()  # important! This enables embedding dropout for classifier-free guidance

    # Variables for monitoring/logging purposes:
    train_steps = 0
    log_steps = 0
    running_loss = 0
    running_mse = 0
    running_mp = 0
    best_running_mse = 10000
    best_running_mp = 10000
    
    start_time = time()
    logger.info(f"Training for {adjusted_epochs} epochs...")
    for epoch in range(adjusted_epochs):
        # sampler.set_epoch(epoch)
        logger.info(f"Beginning epoch {epoch}...")
        set_random_seed()
        for ii, (x, brainmask, _) in enumerate(loader):
            x = x.to(device)
            with torch.no_grad():
                # Map input images to latent space + normalize latents:
                x = vae.encode(x).sample().mul_(0.18215)
            t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)
 
            mask_patch_size_x = np.random.choice([1,2,4,8], 1, p=[0.2, 0.3, 0.3, 0.2]).item()
            mask_patch_size_y = np.random.choice([1,2,4,8], 1, p=[0.2, 0.3, 0.3, 0.2]).item()
            mask_ratios = np.random.uniform(low=0.0, high=0.4, size = x.shape[0])

            mask = random_mask(x, mask_ratios=mask_ratios, mask_patch_size_x=mask_patch_size_x, mask_patch_size_y=mask_patch_size_y)
            brainmask_reduced = (torch.nn.functional.avg_pool2d(brainmask.unsqueeze(1), 16, 16)<0.9).to(torch.int32)
            brainmask_reduced = torch.repeat_interleave(brainmask_reduced,x.shape[1], 1)
            brainmask_reduced = torch.repeat_interleave(torch.repeat_interleave(brainmask_reduced, 2, 2),2, 3)
            mask[brainmask_reduced==1] = 1

            model_kwargs = {
            'mask': mask
            }

            noise = torch.randn_like(x, device=device)
            loss_dict = diffusion.training_losses(model, x, t, model_kwargs, noise = noise)
            loss = loss_dict["loss"].mean()
            loss.backward()
            
            if (ii + 1) % accumulation_steps == 0:
                opt.step()
                opt.zero_grad() 

            # Log loss values:
            running_loss += loss.item()
            running_mse += loss_dict["mse"].mean().item()
            running_mp += loss_dict["mask_prediction"].mean().item()
            
            log_steps += 1
            train_steps += 1
            if train_steps % args.log_every == 0:
                # Measure training speed:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                # Reduce loss history over all processes:
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                avg_mse = torch.tensor(running_mse / log_steps, device=device)
                avg_mp = torch.tensor(running_mp / log_steps, device=device)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                dist.all_reduce(avg_mse, op=dist.ReduceOp.SUM)
                dist.all_reduce(avg_mp, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / dist.get_world_size()
                avg_mse = avg_mse.item() / dist.get_world_size()
                avg_mp = avg_mp.item() / dist.get_world_size()
                logger.info(f"Training (step={train_steps:07d}) MSE Loss: {avg_mse:.4f}, Mask_pred Loss: {avg_mp:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")
                # Reset monitoring variables:
                running_loss = 0
                running_mse = 0
                running_mp = 0
                log_steps = 0
                start_time = time()

        scheduler.step()
        
        
        running_mse_val = 0
        running_mp_val = 0
        log_steps_val = 0
                
        if epoch % args.ckpt_every == 0 and epoch!=0:
            model.eval()
            logger.info(f"Valdation epoch {epoch}...")
            fix_seed(args.global_seed)
            for ii, (x, brainmask, _) in enumerate(val_loader):
                with torch.no_grad():
                    x = x.to(device)
                
                    # Map input images to latent space + normalize latents:
                    x = vae.encode(x).sample().mul_(0.18215)
                    t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)
        
                    mask_patch_size_x = np.random.choice([1,2,4,8], 1, p=[0.2, 0.3, 0.3, 0.2]).item()
                    mask_patch_size_y = np.random.choice([1,2,4,8], 1, p=[0.2, 0.3, 0.3, 0.2]).item()
                    mask_ratios = np.random.uniform(low=0.0, high=0.3, size = x.shape[0])

                    mask = random_mask(x, mask_ratios=mask_ratios, mask_patch_size_x=mask_patch_size_x, mask_patch_size_y=mask_patch_size_y)
                    brainmask_reduced = (torch.nn.functional.avg_pool2d(brainmask.unsqueeze(1), 16, 16)<0.9).to(torch.int32)
                    brainmask_reduced = torch.repeat_interleave(brainmask_reduced,x.shape[1], 1)
                    brainmask_reduced = torch.repeat_interleave(torch.repeat_interleave(brainmask_reduced, 2, 2),2, 3)
                    mask[brainmask_reduced==1] = 1

                    model_kwargs = {
                    'mask': mask
                    }
                    
                    noise = torch.randn_like(x, device=device)
                    loss_dict = diffusion.training_losses(model, x, t, model_kwargs, noise = noise)
                        

                    # Log loss values:
                    running_mse_val += loss_dict["mse"].mean().item()
                    running_mp_val += loss_dict["mask_prediction"].mean().item()
                    
                    log_steps_val += 1
            
                    # Measure training speed:
                    
            end_time = time()
            steps_per_sec = log_steps_val / (end_time - start_time)
            # Reduce loss history over all processes:
            avg_mse_val = torch.tensor(running_mse_val / log_steps_val, device=device)
            avg_mp_val = torch.tensor(running_mp_val / log_steps_val, device=device)
            
            dist.all_reduce(avg_mse_val, op=dist.ReduceOp.SUM)
            dist.all_reduce(avg_mp_val, op=dist.ReduceOp.SUM)
            
            avg_mse_val = avg_mse_val.item() / dist.get_world_size()
            avg_mp_val = avg_mp_val.item() / dist.get_world_size()
            
            logger.info('*'*50)
            logger.info(f"(VAlidation - MSE Loss: {avg_mse_val:.4f}, Mask_pred Loss: {avg_mp_val:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")
            logger.info('*'*50)
            # Reset monitoring variables:

            log_steps_val = 0
            start_time = time()    
            
            model.train()
            
            if running_mse_val < best_running_mse:
                best_running_mse = running_mse_val
                if rank == 0: 
                    # Save checkpoint:
                    checkpoint = {
                        "model": model.module.state_dict(),
                        # "opt": opt.state_dict(),
                        "args": args
                    }
                    checkpoint_path = f"{checkpoint_dir}/best_mse.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")
                    
            if running_mp_val > best_running_mp:
                best_running_mp = running_mp_val
                if rank == 0: 
                    # Save checkpoint:
                    checkpoint = {
                        "model": model.module.state_dict(),
                        # "opt": opt.state_dict(),
                        "args": args
                    }
                    checkpoint_path = f"{checkpoint_dir}/best_mask_prediction.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")
            
            

            if rank == 0: 
                # Save checkpoint:
                checkpoint = {
                    "model": model.module.state_dict(),
                    # "opt": opt.state_dict(),
                    "args": args
                }
                checkpoint_path = f"{checkpoint_dir}/last.pt"
                torch.save(checkpoint, checkpoint_path)
                logger.info(f"Saved checkpoint to {checkpoint_path}")

                  
            dist.barrier()



    model.eval()  # important! This disables randomized embedding dropout

    logger.info("Done!")
    cleanup()
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_size", type=str, choices=['UNet_XS', 'UNet_S', 'UNet_M', 'UNet_L', 'UNet_XL'], default="UNet_L")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--modality", type=str, choices=['T1', 'T2', 'FLAIR', 'T1CE'], default="T1")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--warmup-epochs", type=int, default=10)
    parser.add_argument("--global-batch-size", type=int, default=96)
    parser.add_argument("--global-seed", type=int, default=10)
    parser.add_argument("--train-data-root", type=str, default="./data/train/")  
    parser.add_argument("--val-data-root", type=str, default="./data/val/")  
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--ckpt-every", type=int, default=10)
    parser.add_argument("--lr", type=float, default=4e-5)
    parser.add_argument("--local-rank", type=int, default=0)
    parser.add_argument("--mask-ratio", type=float, default=0.75)
    parser.add_argument("--augmentation", type=bool, default=False)
    
    args = parser.parse_args()
    args.results_dir = f"results_MAD-AD_{args.modality}_{args.model_size}"
    main(args)

