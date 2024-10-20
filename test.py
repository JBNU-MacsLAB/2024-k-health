import torch
from torch.utils.data import DataLoader
import albumentations as A
import segmentation_models_pytorch as smp
from lib.datasets.dicom_nii_2d_dataset_filter import DicomNii2DDataset
from run import evaluate

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# Parameters
IMG_RESIZE = 256
BATCH_SIZE = 4

# Dataset
transform = A.Compose([
    A.Resize(height=IMG_RESIZE, width=IMG_RESIZE),
    A.Normalize(),
])

test_dataset = DicomNii2DDataset('./20241008_smart_health_care2_abnormal_public_001_200/breast', transform)
test_dataloader = DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2
)

model = smp.MAnet(
    encoder_name='mobilenet_v2',  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
    encoder_weights='imagenet',  # use `imagenet` pre-trained weights for encoder initialization
    in_channels=1,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
    classes=1,  # model output channels (number of classes in your dataset)
)
model = model.to(device)

file = 'loocv_533_model_complete_state_dict_0100.pth'

# state_dict 로드
model.load_state_dict(torch.load(file, map_location=device, weights_only=False))
evaluate(model, test_dataloader, device)
