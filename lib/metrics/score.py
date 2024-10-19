"""
평가 함수 정의
"""

import torch
from torchmetrics.segmentation import GeneralizedDiceScore, MeanIoU


NUM_CLASSES = 2


def get_gds(device) -> GeneralizedDiceScore:
    """
    return GeneralizedDiceScore
    """

    return GeneralizedDiceScore(
        NUM_CLASSES,
        include_background=False,
        per_class=True,
        compute_on_cpu=True,
    ).to(device)


def calculate_gds(preds: torch.Tensor, target: torch.Tensor, device) -> torch.Tensor:
    """
    https://lightning.ai/docs/torchmetrics/stable/segmentation/generalized_dice.html
    """

    metric = get_gds(device)
    result = metric(preds, target)
    # print("gds", result)
    return result


def get_miou(device) -> MeanIoU:
    """
    return MeanIoU
    """

    return MeanIoU(
        NUM_CLASSES,
        include_background=False,
        per_class=True,
        compute_on_cpu=True,
    ).to(device)


def calculate_miou(preds: torch.Tensor, target: torch.Tensor, device) -> torch.Tensor:
    """
    https://lightning.ai/docs/torchmetrics/stable/segmentation/mean_iou.html
    """

    metric = get_miou(device)
    result = metric(preds, target)
    # print("miou", result)
    return result


def evaluate_model(preds: torch.Tensor, masks: torch.Tensor, device):
    """
    preds (Tensor): An one-hot boolean tensor of shape (N, C, ...) with N being the number of samples and C the number of classes.
    target (Tensor): An one-hot boolean tensor of shape (N, C, ...) with N being the number of samples and C the number of classes.
    """

    threshold = 0.75

    # 예측 값 이진화 (시그모이드 함수 적용)
    sigmoid_preds = torch.sigmoid(preds)
    sigmoid_preds = (sigmoid_preds >= threshold).int()

    target = masks.int()

    gds = calculate_gds(sigmoid_preds, target, device)
    miou = calculate_miou(sigmoid_preds, target, device)

    return gds, miou
