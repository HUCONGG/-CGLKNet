# CGLKNet
 CGLKNet is used for medical image segmentation
# Requirements
Python >= 3.8

PyTorch >= 1.9.0

torchvision >= 0.10.0

CUDA 11.1+ (recommended for GPU training)
 #  requirements.txt
 numpy>=1.21.0
opencv-python>=4.5.0
albumentations>=1.1.0
timm>=0.6.0
tqdm>=4.64.0
tensorboard>=2.10.0
scikit-learn>=1.0.0
scipy>=1.7.0
# Install PyTorch with CUDA support
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113

# Install other dependencies
pip install -r requirements.txt
 # dataset
CVC-ClinicDB: Polyp segmentation dataset

GLAS: Gland segmentation dataset

BUSI: Breast ultrasound dataset

ISIC2016: Skin lesion dataset
# dataset

data/
- CVC-ClinicDB/
  - images/
    - 0001.png
    - ...
  - masks/
    - 0001.png
    - ...
- GLAS/
  - images/
    - train/
    - test/
    - ...
  - masks/
    - train/
    - test/
    - ...
- BUSI/
  - images/
  - masks/
- ISIC2016/
  - images/
  - masks/
