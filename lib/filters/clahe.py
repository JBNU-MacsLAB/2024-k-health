import torch
import cv2
import numpy as np


class CLAHE:
    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8)):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
        self.clahe = cv2.createCLAHE(
            clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size
        )

    def apply(self, dicom_tensor):
        dicom_np = dicom_tensor.numpy()

        dicom_np = cv2.normalize(dicom_np, None, 0, 255, cv2.NORM_MINMAX).astype(
            np.uint8
        )

        if dicom_np.ndim == 2:
            clahe_applied = self.clahe.apply(dicom_np)
        else:
            clahe_applied = np.array([self.clahe.apply(slice) for slice in dicom_np])

        clahe_tensor = torch.from_numpy(clahe_applied.astype(np.float32))

        return clahe_tensor
