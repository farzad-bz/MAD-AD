# MAD-AD
This Repository contains the PyTorch implementation of the paper titled: "MAD-AD: Masked Diffusion Autoencoder for Unsupervised Brain Anomaly Detection", Accepted in IPMI 2-25. 

## Setup

### Environment

We utilize the `Python 3.11` interpreter in our experiments. Install the required packages using the following command:
```bash
pip3 install -r requirements.txt
```

### Datasets
Prepare your data by registering to MNI_152_1mm and preprocessing, normalization, and extracting axial slices. Ensure that the training and validation sets consist only of normal, healthy data, while the test set should contain abnormal slices. Organize the files using the following structure:
```
├── Data
    ├── train
    │   ├── brain_scan_{train_image_id}_slice_{slice_idx}_{modality}.png
    │   ├── brain_scan_{train_image_id}_slice_{slice_idx}_brainmask.png
    │   └── ...
    ├── val
    │   ├── brain_scan_{val_image_id}_slice_{slice_idx}_{modality}.png
    │   ├── brain_scan_{val_image_id}_slice_{slice_idx}_brainmask.png
    │   └── ...
    └── test
    │   ├── brain_scan_{test_image_id}_slice_{slice_idx}_{modality}.png
    │   ├── brain_scan_{test_image_id}_slice_{slice_idx}_brainmask.png
    │   ├── brain_scan_{test_image_id}_slice_{slice_idx}_segmentation.png
        └── ...

```

## train and fine-tune VAE

If you want to train your own VAE from the beginning, follow [LDM-VAE](https://github.com/CompVis/latent-diffusion?tab=readme-ov-file#training-autoencoder-models).  Also, we have adapted and finetuned the RGB pre-trained models for 1-channel medical brain images, and we provide access to trained VAE model upon acceptance of the paper. 

## Train

Train MAD-AD with the following command:

```bash
torchrun train_MAD_AD.py \
            --model UNet_L \
            --mask-random-ratio True \
            --image-size 256 \
            --augmentation True \
            --vae_path checkpoints/klf8_medcical.ckpt \
            --train_data_root ./data/train/ \
            --val_data_root ./data/val/ \
            --modality T1 \
            --ckpt-every 20 
```

## Sample Results

![DeCo-Diff](./qualitative-results.png)
