import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
import albumentations as A
import segmentation_models_pytorch as smp
from lib.datasets.dicom_nii_2d_dataset_filter import DicomNii2DDataset
from lib.metrics.score import evaluate_model
from .run import train
import matplotlib.pyplot as plt
import os

graph_dir = './graph'
if not os.path.exists(graph_dir):
    os.makedirs(graph_dir)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# Parameters
IMG_RESIZE = 256
BATCH_SIZE = 64
num_epochs = 100
learning_rate = 1e-3

# Dataset
transform = A.Compose([
    A.Resize(height=IMG_RESIZE, width=IMG_RESIZE),
    A.Normalize(),
])

dataset = DicomNii2DDataset('./20241008_smart_health_care2_abnormal_public_001_200/breast', transform)

# LOOCV
n_samples = len(dataset)
loocv_scores = []
best_score = float('-inf')
best_model_state = None
best_test_img = 0

def plot_prediction_overlay(input_image, true_mask, pred_mask, save_path):
    fig, ax = plt.subplots(figsize=(5, 5))

    # Input image
    ax.imshow(input_image.squeeze(), cmap='gray')
    ax.set_title('Overlay of Masks on Input Image')
    ax.axis('off')

    # Predicted mask
    ax.imshow(pred_mask.squeeze(), alpha=0.65, cmap='Blues')
    ax.imshow(true_mask.squeeze(), alpha=0.5, cmap='Reds')

    # Save the image
    plt.savefig(save_path)
    plt.close()

with open('train_log.txt', 'w') as log_file:
    for i in range(n_samples):
        # Create train and validation indices
        train_indices = list(range(n_samples))
        val_index = train_indices.pop(i)

        # Create samplers for train and validation splits
        train_sampler = SubsetRandomSampler(train_indices)
        val_sampler = SubsetRandomSampler([val_index])

        # Create data loaders
        train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=train_sampler, num_workers=2)
        val_loader = DataLoader(dataset, batch_size=1, sampler=val_sampler, num_workers=2)

        model = smp.MAnet(
            encoder_name='mobilenet_v2',  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights='imagenet',  # use `imagenet` pre-trained weights for encoder initialization
            in_channels=1,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=1,  # model output channels (number of classes in your dataset)
        )
        model = model.to(device)

        # Train the model
        train_losses, train_gds, train_miou = train(model, num_epochs, learning_rate, train_loader, device)

        # Validate the model
        model.eval()
        val_gds = []
        val_miou = []
        with torch.no_grad():
            for v, (inputs, masks) in enumerate(val_loader):
                inputs, masks = inputs.to(device), masks.to(device)

                # Add extra dimension if necessary
                if inputs.dim() == 3:
                    inputs = inputs.unsqueeze(0)  # Add batch dimension
                if masks.dim() == 3:
                    masks = masks.unsqueeze(0)  # Add batch dimension

                outputs = model(inputs)
                gds, miou = evaluate_model(outputs, masks, device)
                score = gds + miou

                val_gds.append(gds.cpu().numpy())
                val_miou.append(miou.cpu().numpy())

        avg_val_gds = sum(val_gds) / len(val_gds)
        avg_val_miou = sum(val_miou) / len(val_miou)
        avg_val_score = avg_val_gds + avg_val_miou

        print(f'[VAL] LOOCV Fold {i + 1}/{n_samples} | GDS: {avg_val_gds}, mIoU: {avg_val_miou}, score(GDS + mIoU): {avg_val_score}')
        log_file.write(f'[VAL] LOOCV Fold {i + 1}/{n_samples} | GDS: {avg_val_gds}, mIoU: {avg_val_miou}, score(GDS + mIoU): {avg_val_score}\n')
        log_file.flush()

        # Save the best model
        if avg_val_score > best_score:
            best_score = avg_val_score
            best_model_state = model.state_dict()
            best_test_img = i + 1
            print(f'===> LOOCV Fold {i+1}, New best score: {best_score}!')

            plot_prediction_overlay(
                inputs.cpu().numpy()[0, 0],
                masks.cpu().numpy()[0, 0],
                outputs.cpu().numpy()[0, 0],
                os.path.join(graph_dir, f'loocv_fold_{i + 1}_sample_{v + 1}.png')
            )

            # Save the best model
            team = '533'
            torch.save(
                best_model_state,
                f'loocv_{team}_model_complete_state_dict_{num_epochs:04}.pth',
            )

            plt.figure(figsize=(10, 6))
            epochs = list(range(1, len(train_losses) + 1))

            plt.plot(epochs, train_losses, label='Train Loss')
            plt.plot(epochs, [gds.cpu().numpy() for gds in train_gds], label='Train GDS')
            plt.plot(epochs, [miou.cpu().numpy() for miou in train_miou], label='Train mIoU')

            plt.xlabel('Epochs')
            plt.ylabel('Metrics')
            plt.title(f'LOOCV Fold {i + 1}')
            plt.legend()

            plt.savefig(os.path.join(graph_dir, f'LOOCV_Fold_{i + 1}.png'))
            plt.close()

    print(f'when testing with image {best_test_img}, we got best score!')
    log_file.write(f'LOOCV Fold {best_test_img} got best score: {best_score}!\n')
    log_file.flush()
