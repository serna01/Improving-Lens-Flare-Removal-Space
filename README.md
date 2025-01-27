
# Lens Flare Removal Models: A Comparative Guide

This repository contains a detailed implementation and comparison of three different lens flare removal models:
- **7Kpp**: Based on Flare7K++ dataset.
- **Google Research**: Published in ICCV 2021.
- **Improved Lens Flare Removal (ILF)**: ICCV 2023.

## Table of Contents
- [Overview](#overview)
- [Models](#models)
- [Setup](#setup)
- [Running the Models](#running-the-models)
- [Training](#training)
- [Notes on Jetson Nano Orin](#notes-on-jetson-nano-orin)
- [Acknowledgements](#acknowledgements)

---

## Overview

This project evaluates lens flare removal models, providing scripts for running pre-trained models and training new ones. All models are tested in Python virtual environments for dependency isolation. NVIDIA CUDA support is required for optimal performance.

## Models

| Model                | Paper                                                                                   | Github Repository                                                            |
|----------------------|-----------------------------------------------------------------------------------------|------------------------------------------------------------------------------|
| **7Kpp**             | Flare7K++: Mixing Synthetic and Real Datasets for Nighttime Flare Removal and Beyond   | [Link](https://github.com/ykdai/Flare7K.git)                                |
| **Google Research**  | How to train neural networks for flare removal (ICCV 2021)                             | [Link](https://github.com/google-research/google-research/tree/master/flare_removal) |
| **ILF**              | Improving Lens Flare Removal (ICCV 2023)                                               | [Link](https://github.com/YuyanZhou1/Improving-Lens-Flare-Removal)          |

## Setup

1. **Create a Python Virtual Environment for each model**:
    ```bash
    python3 -m venv env
    source env/bin/activate  # On Windows, use env\Scripts\activate
    ```
2. **Install NVIDIA CUDA (if required)**:
    - Ensure compatibility with your GPU and driver version.
    - Export CUDA paths as needed, e.g.,:
      ```bash
      export PATH=/usr/local/cuda-12.5/bin:$PATH
      ```

## Running the Models

### Flare7K++
Run the following command to test:
```bash
python test_large.py --input test/test_images/ --output result --model_path experiments/net_g_last.pth --flare7kpp
```

For custom models:
```bash
python test_large.py --input test/test_images/ --output result/custom --model_path experiments/custom_model.pth --flare7kpp
```

### Google Research
Install TensorFlow with CUDA support:
```bash
pip install tensorflow[and-cuda]==2.15.1
python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

Run inference:
```bash
python3 -m flare_removal.python.remove_flare --ckpt=../train/model --input_dir=../test/test_images --out_dir=../results
```

### ILF
Run inference with the pre-trained model:
```bash
python3 remove_flare.py --input_dir=input_data/LL_lensflare --out_dir=results --model=Uformer --batch_size=2 --ckpt=experiments/trained_model
```

For custom models:
```bash
python3 remove_flare.py --input_dir=input_data/LL_lensflare --out_dir=results/custom --model=Uformer --batch_size=2 --ckpt=experiments/custom_trained_model
```

### Custom Training with Simulation Pipeline

For custom pipelines using simulated scenes with pre-added flares (e.g., `scene_with_flare`), use the following scripts and commands:

#### New Script: **SynthesisV2.py**

This script synthesizes flare-affected images and prepares datasets.

#### Training Command

```bash
python trainV2.py --flare_dir=test_data/batch_003/flares640 --scene_dir=test_data/batch_003/gt640
```

#### Flare Removal

```bash
python3 remove_flare.py \
  --input_dir=input_data \
  --out_dir=results/blender_modelv15102024 \
  --model=Uformer \
  --batch_size=2 \
  --ckpt=experiments/blender_model/train/model
```

### Notes on Dataset Preparation

In the `data_provider.py` code, image shapes for scene and flare datasets are specified:

- **Scene Dataset** (`get_scene_dataset`):
  - Default input shape: `(640, 640, 3)`
  - Scene images are expected to be 640x640 pixels.
- **Flare Dataset** (`get_flare_dataset` and `get_flare_dataset2`):
  - Default input shape: `(752, 1008, 3)`
  - Flare images are expected to be 752x1008 pixels.

Resize images as needed for compatibility.


## Training

### Flare7K++
Train the model with a baseline configuration:
```bash
python basicsr/train.py -opt options/uformer_flare7kpp_baseline_option.yml
```

### Google Research
Train the model with custom data:
```bash
python3 -m flare_removal.python.train --train_dir=../train/ --scene_dir=../scene_data/ --flare_dir=../flare_data/
```

### ILF
Train with preprocessed datasets:
```bash
python train.py --flarec_dir=path/to/captured_flare --flares_dir=path/to/simulated_flare --scene_dir=path/to/scene_image
```

## Notes on Jetson Nano Orin

- **Flare7K++** requires PyTorch. Install using NVIDIA's package manager:
  - [PyTorch JetPack](https://forums.developer.nvidia.com/t/torchvision-version-jetpack-6-0/301709/2)
- **Google Research** requires TensorFlow:
  ```bash
  sudo apt-get install libhdf5-serial-dev hdf5-tools libhdf5-dev
  pip3 install --no-cache https://developer.download.nvidia.com/compute/redist/jp/v60/tensorflow/tensorflow-2.15.0+nv24.05-cp310-cp310-linux_aarch64.whl
  ```

## Acknowledgements
- [Flare7K++](https://github.com/ykdai/Flare7K.git)
- [Google Research Flare Removal](https://github.com/google-research/google-research/tree/master/flare_removal)
- [Improved Lens Flare Removal (ICCV 2023)](https://github.com/YuyanZhou1/Improving-Lens-Flare-Removal)

---
For further details or contributions, feel free to submit an issue or pull request.
