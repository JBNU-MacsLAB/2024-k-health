# Team 533

## How to Use
0. put dataset dir
1. activate conda environment
   1. ```conda create -n <env_name> python=3.10```
   2. ```conda activate <env_name>```
2. set environment &rarr; ```pip install -r requirements.txt```
3. model train &rarr; ```python train.py```
    - model: MA-Net(https://ieeexplore.ieee.org/abstract/document/9201310)
    - encoder: MobileNet V2(https://arxiv.org/abs/1801.04381)
    - encoder weights: ImageNet(https://ieeexplore.ieee.org/document/5206848)
4. test &rarr; ```python test.py```

## Best Model

```
BATCH_SIZE = 64
num_epochs = 100
learning_rate = 1e-3
```

LOOCV Fold 115/200
- val GDS = 0.9488313794136047
- val mIoU = 0.9026442170143127
- val score(GDS + mIoU) = 1.8514755964279175

## Structure

```
2024-k-health
├── 20241008_smart_health_care2_abnormal_public_001_200(drop after downloading the dataset)
│   └── breast
│       ├── image
│       │   └── ...
│       └── label
│           └── ...
├── graph(automatically generated when model training starts)
├── lib
│   ├── datasets
│   │   └── dicom_nii_2d_dataset_filter.py
│   ├── filters
│   │   ├── __init__.py
│   │   ├── clahe.py
│   │   └── flip.py
│   ├── losses
│   │   ├── __init__.py
│   │   └── dice_bce.py
│   └── metrics
│       └── score.py
├── loocv_533_model_complete_state_dict_0100.pth(automatically generated while model training)
├── README.md
├── requirements.txt
├── run.py
├── test.py
├── train.py
└── train_log.txt(automatically generated when model training starts)
```
