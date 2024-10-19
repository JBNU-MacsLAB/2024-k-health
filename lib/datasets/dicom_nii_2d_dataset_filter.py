"""
CustomDataset
"""

import os
from typing import Optional

import pydicom
import nibabel as nib
import torch
from torch.utils.data import Dataset
import numpy as np
from albumentations import Compose
from lib.filters import clahe, flip_left

class DicomNii2DDataset(Dataset):
    """
    DicomNii2DDataset 클래스는 DICOM 파일과 NIfTI 파일을 읽어와 2D 이미지와 라벨을 반환하는 PyTorch Dataset입니다.
    """

    def __init__(
        self,
        root_path: str,  # root_path: DICOM 및 NIfTI 파일이 저장된 루트 경로
        transform: (
                Optional[Compose]
        ) = None,  # transform: 데이터 증강에 사용할 albumentations 변환 (선택적)
    ):
        self.dicom_dir = os.path.join(root_path, "image/abnormal")
        self.nii_dir = os.path.join(root_path, "label/nii")
        self.transform = transform

        # DICOM 파일 목록을 정렬하여 가져옴
        self.dicom_filenames = sorted(
            [f for f in os.listdir(self.dicom_dir) if f.endswith(".dcm")]
        )
        # NIfTI 파일 목록을 정렬하여 가져옴
        self.nii_filenames = sorted(
            [f for f in os.listdir(self.nii_dir) if f.endswith(".nii")]
        )

        # 파일 이름에서 확장자를 제거한 리스트 생성
        dicom_base_names = [os.path.splitext(f)[0] for f in self.dicom_filenames]
        nii_base_names = [os.path.splitext(f)[0] for f in self.nii_filenames]

        # DICOM 파일과 NIfTI 파일의 이름이 일치하는지 확인
        assert (
            dicom_base_names == nii_base_names
        ), "DICOM 파일과 NIfTI 파일의 이름이 매칭되지 않습니다."

        # 공통된 파일 이름만 저장 (확장자 제거)
        self.common_filenames = sorted(
            [os.path.splitext(f)[0] for f in self.dicom_filenames]
        )

    def __len__(self):
        # 데이터셋의 총 길이 (DICOM 파일의 수)를 반환
        return len(self.dicom_filenames)

    def __getitem__(self, idx):
        # 주어진 인덱스에 해당하는 DICOM 및 NIfTI 파일을 읽어와 처리 후 반환

        # 인덱스에 해당하는 공통 파일 이름 가져오기
        base_name = self.common_filenames[idx]

        dicom_path = os.path.join(self.dicom_dir, base_name + ".dcm")
        nii_path = os.path.join(self.nii_dir, base_name + ".nii")

        # DICOM 파일 읽기
        dicom_image = pydicom.dcmread(dicom_path)
        dicom_array = dicom_image.pixel_array  # DICOM 이미지의 픽셀 배열 가져오기
        dicom_array = dicom_array.astype(np.float32)

        # NIfTI 파일 읽기
        nii_image = nib.nifti1.load(nii_path)
        nii_array = nii_image.get_fdata()  # NIfTI 이미지의 데이터를 배열로 가져오기
        # NIfTI 라벨 값을 0 또는 1로 변환 (라벨 이진화)
        nii_array = np.where(nii_array > 0, 1, 0)
        nii_array = nii_array.astype(np.float32)

        # NIfTI 이미지를 회전하고 좌우 반전 (맞춤을 위해)
        nii_array = np.rot90(nii_array, k=-1)
        nii_array = np.fliplr(nii_array)

        # 데이터 증강이 설정되어 있을 경우, 이미지와 마스크에 적용
        if self.transform:
            augmentations = self.transform(image=dicom_array, mask=nii_array)
            dicom_array = augmentations["image"]
            nii_array = augmentations["mask"]

        # numpy 배열을 PyTorch 텐서로 변환
        dicom_tensor = torch.from_numpy(dicom_array)
        nii_tensor = torch.from_numpy(nii_array)

        # Filter -----------------------------
        dicom_tensor, flip_applied = flip_left(dicom_tensor) # Flip Left

        if flip_applied:
            nii_tensor = torch.from_numpy(np.fliplr(nii_tensor.numpy()).copy())

        dicom_tensor = clahe(dicom_tensor) # CLAHE Filter

        # DICOM 이미지 텐서와 NIfTI 마스크 텐서를 반환
        return dicom_tensor, nii_tensor
