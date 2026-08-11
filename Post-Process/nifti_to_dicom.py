# -*- coding: utf-8 -*-
# Convert NIfTI (.nii / .nii.gz) volumes -> a DICOM series per patient, in the
# same layout the pipeline consumes: <output>/<patient>/{cbct,ct,mask}/slice_xxxx.dcm
#
# Expected input layout:
#   <brain_root>/<patient>/cbct.nii.gz   (or cbct.nii)
#   <brain_root>/<patient>/ct.nii.gz
#   <brain_root>/<patient>/mask.nii.gz
# If <brain_root> itself holds the .nii(.gz) files (no patient subfolders), it is
# treated as ONE patient whose ID = the folder name.
#
# HU is preserved on round-trip: stored(uint16) = round(HU - intercept),
# intercept = floor(min HU). Reading back gives pixel*1 + intercept = HU.
# NOTE: the intercept is per-volume (= its own min) -> this is exactly why the
# resulting DICOM has a different RescaleIntercept per center (2BA/2BB/2BC), and
# why datasets.py forces a common CBCT intercept.
#
# Usage:
#   python nifti_to_dicom.py <brain_root> <output_root>
#   python nifti_to_dicom.py            # uses the defaults below
import os
import sys
import glob
import argparse
import datetime
import numpy as np
import nibabel as nib
import pydicom
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import generate_uid, ExplicitVRLittleEndian

# ===================== DEFAULTS (overridable via CLI) =====================
DEFAULT_BRAIN_ROOT  = "./nii"              # folder with <patient>/*.nii.gz
DEFAULT_OUTPUT_ROOT = "./dicom_from_nii"   # set to "../dataset" to feed the pipeline directly
# =========================================================================

UINT16_MAX = 65535


def nifti_to_dicom(nifti_path, output_dir, modality="CT", is_mask=False, series_description=""):
    os.makedirs(output_dir, exist_ok=True)

    nii = nib.load(nifti_path)          # handles both .nii and .nii.gz
    volume = nii.get_fdata()

    if is_mask:
        rescale_intercept = 0
        volume = np.clip(np.round(volume), 0, UINT16_MAX).astype(np.uint16)
    else:
        # floor(min) guarantees (HU - intercept) >= 0 even when min is negative &
        # non-integer -> avoids uint16 underflow/overflow (the original int(min) bug).
        rescale_intercept = int(np.floor(float(volume.min())))
        vol = np.round(volume - rescale_intercept)
        volume = np.clip(vol, 0, UINT16_MAX).astype(np.uint16)

    H, W, n_slices = volume.shape
    zooms = nii.header.get_zooms()
    pixel_spacing   = [float(zooms[0]), float(zooms[1])]
    slice_thickness = float(zooms[2]) if len(zooms) > 2 else 1.0

    series_instance_uid = generate_uid()
    study_instance_uid  = generate_uid()
    study_date = datetime.datetime.now().strftime("%Y%m%d")
    study_time = datetime.datetime.now().strftime("%H%M%S")
    patient_id = os.path.basename(os.path.dirname(output_dir))

    for i in range(n_slices):
        slice_data = volume[:, :, i].astype(np.uint16)

        file_meta = FileMetaDataset()
        file_meta.MediaStorageSOPClassUID    = "1.2.840.10008.5.1.4.1.1.2"
        file_meta.MediaStorageSOPInstanceUID = generate_uid()
        file_meta.TransferSyntaxUID          = ExplicitVRLittleEndian
        file_meta.ImplementationClassUID     = generate_uid()

        ds = FileDataset(None, {}, file_meta=file_meta, preamble=b"\0" * 128)

        ds.PatientName      = "Anonymous"
        ds.PatientID        = patient_id
        ds.StudyInstanceUID = study_instance_uid
        ds.StudyDate        = study_date
        ds.StudyTime        = study_time
        ds.AccessionNumber  = ""

        ds.SeriesInstanceUID = series_instance_uid
        ds.SeriesNumber      = 1
        ds.Modality          = modality
        ds.SeriesDescription = series_description        # "cbct", "ct", or "mask"
        ds.StudyDescription  = "Brain CBCT2CT"   # free-form description
        ds.PatientPosition   = "HFS"

        ds.SOPClassUID    = file_meta.MediaStorageSOPClassUID
        ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
        ds.InstanceNumber = i + 1

        ds.Rows    = H
        ds.Columns = W
        ds.PixelSpacing            = pixel_spacing
        ds.SliceThickness          = slice_thickness
        ds.ImagePositionPatient    = [0.0, 0.0, float(i) * slice_thickness]
        ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
        ds.SliceLocation           = float(i) * slice_thickness

        ds.SamplesPerPixel           = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.BitsAllocated             = 16
        ds.BitsStored                = 16
        ds.HighBit                   = 15
        ds.PixelRepresentation       = 0
        ds.RescaleSlope              = 1
        ds.RescaleIntercept          = rescale_intercept

        # the correct way to set pixel data in pydicom
        ds.PixelData = slice_data.tobytes()
        ds['PixelData'].VR = 'OB'   # main fix

        out_path = os.path.join(output_dir, f"slice_{i+1:04d}.dcm")
        pydicom.dcmwrite(out_path, ds, write_like_original=False)

    print(f"  [ok] {series_description or modality}: {n_slices} slices "
          f"(intercept={rescale_intercept}) -> {output_dir}")


def find_modality_file(patient_dir, key):
    """Locate a modality NIfTI (.nii/.nii.gz). Exact name first, then a glob fallback.
    For 'ct', names containing 'cbct' are excluded so cbct.nii.gz is not picked."""
    for ext in (".nii.gz", ".nii"):
        exact = os.path.join(patient_dir, key + ext)
        if os.path.exists(exact):
            return exact
    hits = sorted(glob.glob(os.path.join(patient_dir, f"*{key}*.nii*")))
    if key == "ct":
        hits = [h for h in hits if "cbct" not in os.path.basename(h).lower()]
    return hits[0] if hits else None


def convert_patient(patient_dir, output_root):
    patient_id  = os.path.basename(patient_dir.rstrip("/\\"))
    patient_out = os.path.join(output_root, patient_id)

    # key -> (output folder, DICOM Modality, is_mask)
    modality_map = {
        "cbct": ("cbct", "CT", False),
        "ct":   ("ct",   "CT", False),
        "mask": ("mask", "CT", True),
    }

    print(f"\n[Patient ID] {patient_id}")
    found_any = False
    for key, (folder_name, modality, is_mask) in modality_map.items():
        nifti_path = find_modality_file(patient_dir, key)
        if nifti_path is None:
            print(f"  [skip] {key}.nii(.gz) not found.")
            continue
        found_any = True
        output_dir = os.path.join(patient_out, folder_name)
        nifti_to_dicom(nifti_path, output_dir, modality=modality,
                       is_mask=is_mask, series_description=folder_name)
    return found_any


def convert_all(brain_root, output_root):
    if not os.path.isdir(brain_root):
        print(f"brain_root does not exist: {brain_root}")
        return

    patient_dirs = sorted(d for d in glob.glob(os.path.join(brain_root, "*")) if os.path.isdir(d))

    # flat mode: no subfolders but .nii(.gz) sit directly in brain_root -> one patient
    if not patient_dirs and glob.glob(os.path.join(brain_root, "*.nii*")):
        patient_dirs = [brain_root.rstrip("/\\")]

    if not patient_dirs:
        print(f"No patient subfolders and no .nii(.gz) files found in '{brain_root}'")
        return

    print(f"Found {len(patient_dirs)} patient(s). Output -> {output_root}/")
    n_ok = 0
    for d in patient_dirs:
        n_ok += 1 if convert_patient(d, output_root) else 0

    print(f"\nDone! {n_ok}/{len(patient_dirs)} patient(s) converted. Saved at: {output_root}/")


def main():
    ap = argparse.ArgumentParser(description="Convert NIfTI volumes to a DICOM series per patient.")
    ap.add_argument("brain_root", nargs="?", default=DEFAULT_BRAIN_ROOT,
                    help=f"folder with <patient>/*.nii(.gz)  (default: {DEFAULT_BRAIN_ROOT})")
    ap.add_argument("output_root", nargs="?", default=DEFAULT_OUTPUT_ROOT,
                    help=f"where to write the DICOM tree  (default: {DEFAULT_OUTPUT_ROOT})")
    args = ap.parse_args()
    convert_all(brain_root=args.brain_root, output_root=args.output_root)


if __name__ == "__main__":
    main()
