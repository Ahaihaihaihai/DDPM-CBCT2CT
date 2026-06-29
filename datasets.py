# -*- coding: utf-8 -*-
import os
import glob
import torch
import torch.nn.functional as F
import numpy as np
import pydicom
from torch.utils.data import Dataset


# ============================================================================
# EXPLICIT SPLIT (per team request) -- based on patient ID.
# ============================================================================
TEST_PATIENTS = [
    '2BA007', '2BA014', '2BA028', '2BA035', '2BA049', '2BA056', '2BA063',
    '2BA077', '2BB007', '2BB011', '2BB017', '2BB028', '2BB049', '2BB075',
    '2BB109', '2BB151', '2BB175', '2BB179', '2BC007', '2BC014', '2BC021',
    '2BC042', '2BC070', '2BC084',
]
VAL_PATIENTS = [
    '2BA037', '2BA044', '2BA088', '2BB034', '2BB083', '2BB096', '2BB198',
    '2BC016', '2BC027', '2BC037', '2BC056', '2BC081',
]


class BrainDataset(Dataset):
    """
    DICOM folder structure:
        brain_DICOM/2BA007/{cbct,ct,mask}/slice_xxxx.dcm

    CBCT HU CALIBRATION CORRECTION (IMPORTANT):
      CBCT RescaleIntercept is inconsistent across patient series
      (e.g. 2BA=0, 2BB=-1000, 2BC=-1024), whereas CT is consistently -1024.
      As a result CBCT 2BA is shifted by +1024 HU (air reads 0, not -1000).
      To align the CBCT HU scale with CT across all patients, the CBCT intercept
      is FORCED to `cbct_intercept` (default -1024), IGNORING the wrong tag.
      CT keeps the intercept from its tag (already consistent).
    """

    HU_MIN = -1000
    HU_MAX = 2000

    def __init__(
        self,
        root,
        mode="train",
        target_size=(256, 256),
        test_max_slices=None,
        apply_mask_to_image=True,
        cbct_intercept=-1024.0,   # <-- CORRECTION: force CBCT intercept (None = use original tag)
    ):
        self.root = root
        self.mode = mode
        self.target_size = target_size
        self.test_max_slices = test_max_slices
        self.apply_mask_to_image = apply_mask_to_image
        self.cbct_intercept = cbct_intercept

        self.slices = []

        patient_dirs = sorted([
            d for d in glob.glob(os.path.join(root, "*"))
            if os.path.isdir(d)
        ])
        if len(patient_dirs) == 0:
            raise RuntimeError(f"No patient folders found in: {root}")

        test_set = set(TEST_PATIENTS)
        val_set  = set(VAL_PATIENTS)

        if mode == "train":
            selected = [d for d in patient_dirs
                        if os.path.basename(d) not in test_set
                        and os.path.basename(d) not in val_set]
        elif mode == "val":
            selected = [d for d in patient_dirs if os.path.basename(d) in val_set]
        elif mode == "test":
            selected = [d for d in patient_dirs if os.path.basename(d) in test_set]
        else:
            raise ValueError(f"mode must be 'train', 'val', or 'test'. Got: {mode}")

        found_ids = {os.path.basename(d) for d in patient_dirs}
        if mode == "test":
            missing = test_set - found_ids
            if missing:
                print(f"[WARN] test IDs not found in folder: {sorted(missing)}")
        if mode == "val":
            missing = val_set - found_ids
            if missing:
                print(f"[WARN] val IDs not found in folder: {sorted(missing)}")

        skipped = 0
        for patient_dir in selected:
            cbct_dir = os.path.join(patient_dir, "cbct")
            ct_dir   = os.path.join(patient_dir, "ct")
            mask_dir = os.path.join(patient_dir, "mask")

            if not (os.path.isdir(cbct_dir) and os.path.isdir(ct_dir) and os.path.isdir(mask_dir)):
                print(f"[WARN] cbct/ct/mask folders incomplete, skipped: {patient_dir}")
                skipped += 1
                continue

            cbct_files = sorted(glob.glob(os.path.join(cbct_dir, "*.dcm")))
            ct_files   = sorted(glob.glob(os.path.join(ct_dir, "*.dcm")))
            mask_files = sorted(glob.glob(os.path.join(mask_dir, "*.dcm")))

            if not (len(cbct_files) == len(ct_files) == len(mask_files)) or len(cbct_files) == 0:
                print(f"[WARN] Slice count mismatch / empty in {patient_dir}: "
                      f"CBCT={len(cbct_files)}, CT={len(ct_files)}, MASK={len(mask_files)} - skipped")
                skipped += 1
                continue

            for cbct_f, ct_f, mask_f in zip(cbct_files, ct_files, mask_files):
                self.slices.append((cbct_f, ct_f, mask_f))

        if mode == "test" and test_max_slices is not None:
            self.slices = self.slices[:test_max_slices]

        print(f"[{mode}] Selected patients: {len(selected) - skipped} valid, "
              f"{skipped} skipped | Total slices: {len(self.slices)} | "
              f"apply_mask_to_image={self.apply_mask_to_image} | "
              f"cbct_intercept={self.cbct_intercept}")

        if len(self.slices) == 0:
            raise RuntimeError(
                f"No valid slices for mode='{mode}'. "
                f"Check whether the patient folder names match the ID list."
            )

    @staticmethod
    def _load_dicom_array(path, force_intercept=None):
        dcm = pydicom.dcmread(path)
        arr = dcm.pixel_array.astype(np.float32)
        slope = float(getattr(dcm, "RescaleSlope", 1.0))
        if force_intercept is not None:
            intercept = float(force_intercept)        # force (calibration correction)
        else:
            intercept = float(getattr(dcm, "RescaleIntercept", 0.0))
        return arr * slope + intercept

    @staticmethod
    def _load_dicom_mask(path):
        dcm = pydicom.dcmread(path)
        return dcm.pixel_array.astype(np.float32)

    @classmethod
    def _normalize_hu(cls, arr):
        arr = np.clip(arr, cls.HU_MIN, cls.HU_MAX)
        arr = (arr - cls.HU_MIN) / (cls.HU_MAX - cls.HU_MIN)
        return (arr * 2.0 - 1.0).astype(np.float32)

    def _resize_2d(self, tensor, mode="bilinear"):
        tensor = tensor.unsqueeze(0).float()
        if mode == "bilinear":
            tensor = F.interpolate(tensor, size=self.target_size, mode=mode, align_corners=False)
        else:
            tensor = F.interpolate(tensor, size=self.target_size, mode=mode)
        return tensor.squeeze(0)

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, index):
        cbct_path, ct_path, mask_path = self.slices[index]

        # patient ID & slice name from path (.../<patient>/cbct/<slice>.dcm)
        patient_id = os.path.basename(os.path.dirname(os.path.dirname(cbct_path)))
        slice_name = os.path.splitext(os.path.basename(cbct_path))[0]

        # CORRECTION: CBCT uses the forced intercept (-1024); CT uses its original tag
        cbct_arr = self._normalize_hu(
            self._load_dicom_array(cbct_path, force_intercept=self.cbct_intercept))
        ct_arr   = self._normalize_hu(
            self._load_dicom_array(ct_path))
        mask_arr = (self._load_dicom_mask(mask_path) > 0).astype(np.float32)

        if self.apply_mask_to_image:
            cbct_arr = cbct_arr * mask_arr + (-1.0) * (1.0 - mask_arr)
            ct_arr   = ct_arr   * mask_arr + (-1.0) * (1.0 - mask_arr)

        cbct_t = self._resize_2d(torch.from_numpy(cbct_arr).unsqueeze(0), mode="bilinear")
        ct_t   = self._resize_2d(torch.from_numpy(ct_arr).unsqueeze(0),   mode="bilinear")
        mask_t = self._resize_2d(torch.from_numpy(mask_arr).unsqueeze(0), mode="nearest")
        mask_t = (mask_t > 0).float()

        return {
            "CBCT": cbct_t, "pCT": ct_t, "mask": mask_t,
            "patient": patient_id, "slice_name": slice_name,
        }