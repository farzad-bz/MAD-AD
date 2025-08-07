# CUDA_VISIBLE_DEVICES=0 torchrun train-MAD-AD.py \
#             --model UNet_XS \
#             --image-size 256 \
#             --modality FLAIR \
#             --data-root /home-2/ar94660/Datasets/Medical/Brats2023-slices/ \
#             --augmentation True \
#             --ckpt-every 5 \
#             --warmup-epochs 1 

CUDA_VISIBLE_DEVICES=0 torchrun evaluate-MAD-AD.py \
            --data-root /home-2/ar94660/Datasets/Medical/Brats2023-slices/ \
            --model-path /home-2/ar94660/MAD-AD/MAD-AD_FLAIR_UNet_XS/001-UNet_XS/checkpoints/best_mse.pt
