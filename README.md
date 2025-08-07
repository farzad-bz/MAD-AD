
# ✨ MAD-AD ✨
**A PyTorch Implementation for Unsupervised Brain Anomaly Detection**

This repository hosts the official PyTorch implementation for our paper accepted in IPMI2025:  
["MAD-AD: Masked Diffusion for Unsupervised Brain Anomaly Detection"](https://arxiv.org/abs/2502.16943).

---

## 🎨 Approach

![MAD-AD Method](./assets/method.png)

---

## ⚙️ Setup

### 🛠️ Environment

Our experiments run on **Python 3.11**. Install all the required packages by executing:

```bash
pip3 install -r requirements.txt
```

### 📁 Datasets

Prepare your data as follows:

1. **Data Registration & Preprocessing:**  
   - Register with MNI_152_1mm.
   - Preprocess, normalize, and extract axial slices.

2. **Dataset Organization:**  
   - Ensure **training** and **validation** sets contain only normal, healthy data.
   - **Test** set should include abnormal slices.
   - Organize your files using this structure:

   ```
   ├── Data
       ├── train
       │   ├── {train_image_id}-slice_{slice_idx}-{modality}.png
       │   ├── {train_image_id}-slice_{slice_idx}-brainmask.png
       │   └── ...
       ├── val
       │   ├── {val_image_id}-slice_{slice_idx}-{modality}.png
       │   ├── {val_image_id}-slice_{slice_idx}-brainmask.png
       │   └── ...
       └── test
           ├── {test_image_id}-slice_{slice_idx}-{modality}.png
           ├── {test_image_id}-slice_{slice_idx}-brainmask.png
           ├── {test_image_id}-slice_{slice_idx}-segmentation.png
           └── ...
   ```

---

## 🔧 Pretrained Weights & VAE Fine-Tuning

### Pretrained VAE Models

To jumpstart your experiments, we provide pretrained weights adapted for 1-channel medical brain images. These models are available on [HuggingFace](https://huggingface.co/farzadbz/Medical-VAE).

### Train & Fine-Tune VAE

If you prefer to train your own VAE from scratch, please refer to the [LDM-VAE repository](https://github.com/CompVis/latent-diffusion?tab=readme-ov-file#training-autoencoder-models) for detailed instructions.

---

## 🚄 Training MAD-AD

To train MAD-AD, run the following command. This configuration leverages a UNet_L model with data augmentation and integrates the pretrained VAE:

```bash
torchrun train_MAD_AD.py \
            --modality T1 \
            --model UNet_L \
            --mask-ratio 0.75 \
            --image-size 256 \
            --augmentation True \
            --data_root ./data/ \
            --ckpt-every 20 
```


## 🚦 Evaluating MAD-AD
To evaluate MAD-AD model, use the following command.
Note: evaluate-MAD-AD.py loads its configuration and arguments from the YAML file located in the parent directory of the given model checkpoint path. The script computes four evaluation metrics and saves per-image visualizations in the parent folder of the model path:

```bash
torchrun evaluate-MAD-AD.py \
            --data-root ./data/ \
            --model-path ./MAD-AD_T2_UNet_L/001-UNet_L/checkpoints/last.pt
```

---
## 📸 Sample Results


![Sample Results](./assets/results.png)

---

## 📚 Citation & Reference

If you find MAD-AD useful in your research, please cite our work:

```bibtex
@article{beizaee2025mad,
  title={MAD-AD: Masked Diffusion for Unsupervised Brain Anomaly Detection},
  author={Beizaee, Farzad and Lodygensky, Gregory and Desrosiers, Christian and Dolz, Jose},
  journal={arXiv preprint arXiv:2502.16943},
  year={2025}
}
```

