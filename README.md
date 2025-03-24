
# ✨ MAD-AD ✨
**A PyTorch Implementation for Unsupervised Brain Anomaly Detection**

This repository hosts the official PyTorch implementation for our IPMI 2025 paper:  
**der for Unsupervised Brain Anomaly Detection**.

---

## 🎨 Approach


![DeCo-Diff](./assets/method.png)

---

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

If you want to train your own VAE from the beginning, follow [LDM-VAE](https://github.com/CompVis/latent-diffusion?tab=readme-ov-file#training-autoencoder-models).  Also, we have adapted and finetuned the RGB pre-trained models for 1-channel medical brain images,  and now it can be accessed through [HuggingFace!](https://huggingface.co/farzadbz/Medical-VAE)

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


## Citation & Reference

If you use these models in your research, please cite the original latent diffusion work:

```
@article{beizaee2025mad,
  title={MAD-AD: Masked Diffusion for Unsupervised Brain Anomaly Detection},
  author={Beizaee, Farzad and Lodygensky, Gregory and Desrosiers, Christian and Dolz, Jose},
  journal={arXiv preprint arXiv:2502.16943},
  year={2025}
}
```
