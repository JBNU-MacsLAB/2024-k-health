"""
baseline
"""

import torch
from torch import optim
from tqdm import tqdm

from lib.metrics.score import evaluate_model
from lib.losses import DiceBCELoss


def train(model, num_epochs, learning_rate, dataloader, device):
    """
    모델 학습
    """

    # Loss and optimizer
    criterion = DiceBCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_losses = []  # 손실 값
    train_gds = []  # gds
    train_miou = []  # miou

    # Training loop
    for epoch in range(num_epochs):
        # 모델을 학습 모드로 전환
        model.train()

        # 손실 초기화
        running_loss = 0.0

        #
        running_gds = 0.0
        running_miou = 0.0

        for images, masks in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            images = images.float().to(device)
            images = images.unsqueeze(1)  # bhw to b1hw

            masks = masks.float().to(device)
            masks = masks.unsqueeze(1)  # bhw to b1hw

            # 기울기 초기화
            optimizer.zero_grad()

            # 모델 예측
            outputs = model(images)

            # 손실 계산
            loss = criterion(outputs, masks)

            # 역전파
            loss.backward()

            # 가중치 업데이트
            optimizer.step()

            # 손실 누적
            running_loss += loss.item()

            # 점수 계산
            gds, miou = evaluate_model(outputs, masks, device)
            running_gds += gds
            running_miou += miou

        length = len(dataloader)

        # 에포크 손실 출력
        epoch_loss = running_loss / length
        train_losses.append(epoch_loss)

        # gds
        epoch_gds = running_gds / length
        train_gds.append(epoch_gds)

        # miou
        epoch_miou = running_miou / length
        train_miou.append(epoch_miou)

        print(
            f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, GDS: {epoch_gds:.8f}, mIoU: {epoch_miou:.8f}"
        )

    return train_losses, train_gds, train_miou


def evaluate(model, dataloader, device):
    """
    모델 평가
    """

    model.eval()

    #
    inference_gds = 0.0
    inference_miou = 0.0

    with torch.no_grad():
        for images, masks in dataloader:
            images = images.float().to(device)
            images = images.unsqueeze(1)  # bhw to b1hw

            masks = masks.float().to(device)
            masks = masks.unsqueeze(1)  # bhw to b1hw

            # 모델 예측
            outputs = model(images)

            # 점수 계산
            gds, miou = evaluate_model(outputs, masks, device)
            inference_gds += gds
            inference_miou += miou

        length = len(dataloader)

        # gds
        epoch_gds = inference_gds / length

        # miou
        epoch_miou = inference_miou / length

    print(f"GDS: {epoch_gds:.8f}, mIoU: {epoch_miou:.8f}")
