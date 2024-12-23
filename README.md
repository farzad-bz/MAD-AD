# MAD-AD
This Repository contains the PyTorch implementation of the submitted paper titled: "MAD-AD: Masked Diffusion Autoencoder for Unsupervised Brain Anomaly Detection"

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
    │   ├── brain_scan_{train_image_id}_slice_{slice_idx}.png
    │   ├── brain_scan_{train_image_id}_slice_{slice_idx}_brainmask.png
    │   └── ...
    ├── val
    │   ├── brain_scan_{val_image_id}_slice_{slice_idx}.png
    │   ├── brain_scan_{val_image_id}_slice_{slice_idx}_brainmask.png
    │   └── ...
    └── test
    │   ├── brain_scan_{test_image_id}_slice_{slice_idx}.png
    │   ├── brain_scan_{test_image_id}_slice_{slice_idx}_brainmask.png
    │   ├── brain_scan_{test_image_id}_slice_{slice_idx}_segmentation.png
        └── ...

```

## Train

Train our RLR with the following command:

```bash
torchrun train_MAD_AD.py \
            --dataset mvtec \
            --model UNet_L \
            --mask-random-ratio True \
            --object-category all  \
            --image-size 288 \
            --center-size 256 \
            --center-crop True \
            --augmentation True \
            --ckpt-every 20 
```

## Test

Test the model with the following command:

```bash
python evaluate_MAD_AD.py \
            --dataset mvtec \
            --model UNet_L \
            --object-category all  \
            --image-size 288 \
            --center-size 256 \
            --center-crop True \
            --augmentation True \
```
## Sample Results

![DeCo-Diff](./Samples.png)
