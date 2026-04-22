# CGLKNet
 CGLKNet is used for medical image segmentation
 #  requirements.txt
 numpy>=1.21.0
opencv-python>=4.5.0
albumentations>=1.1.0
timm>=0.6.0
tqdm>=4.64.0
tensorboard>=2.10.0
scikit-learn>=1.0.0
scipy>=1.7.0
 # dataset
CVC-ClinicDB: Polyp segmentation dataset

GLAS: Gland segmentation dataset

BUSI: Breast ultrasound dataset

ISIC2016: Skin lesion dataset
data/
├── CVC-ClinicDB/
│   ├── images/
│   │   ├── 0001.png
│   │   └── ...
│   └── masks/
│       ├── 0001.png
│       └── ...
├── GLAS/
│   ├── images/
│   │   ├── train/
│   │   ├── test/
│   │   └── ...
│   └── masks/
│       ├── train/
│       ├── test/
│       └── ...
├── BUSI/
│   ├── images/
│   └── masks/
└── ISIC2016/
    ├── images/
    └── masks/
