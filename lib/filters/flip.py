import torch
import numpy as np

class FlipLeft:
    def __init__(self):
        pass

    def apply(self, dicom_tensor):
        dicom_np = dicom_tensor.numpy()

        left_side_mean = np.mean(dicom_np[:, :dicom_np.shape[1] // 2])
        right_side_mean = np.mean(dicom_np[:, dicom_np.shape[1] // 2:])

        flip_applied = False

        if left_side_mean < right_side_mean:
            dicom_np = np.fliplr(dicom_np)
            flip_applied = True

        flip_applied_tensor = torch.from_numpy(dicom_np.copy())

        return flip_applied_tensor, flip_applied