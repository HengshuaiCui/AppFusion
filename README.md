# AppFusion

## 1. Create Environment
```bash
# Create a new conda environment
conda create -n AppFusion python=3.8 -y

# Activate the environment
conda activate AppFusion
```

## 2. Recommended Requirements

- python = 3.8
- torch = 1.9.1+cu111
- torchvision = 0.10.1+cu111
- timm = 0.9.7
- numpy = 1.21.6
- scipy = 1.7.3
- pillow = 9.5.0
- tqdm = 4.66.1
- tensorboardX = 2.6.2.2
- opencv-python = 4.5.2.54

## 3. Dataset Preparation

We evaluate our method on public infrared–visible image fusion datasets such as **WHU**, **Potsdam**.

Please organize the dataset in the following structure:

```
datasets/
├── train/
│   ├── ir/
│   │   ├── 1.png
│   │   ├── 2.png
│   │   └── ...
│   │
│   └── rgb/
│       ├── 1.png
│       ├── 2.png
│       └── ...
│
├── test/
│   ├── ir/
│   │   ├── 1.png
│   │   ├── 2.png
│   │   └── ...
│   │
│   └── rgb/
│       ├── 1.png
│       ├── 2.png
│       └── ...
```

## 4. Training

Run the following command to train the fusion model:

```bash
python train.py
```
