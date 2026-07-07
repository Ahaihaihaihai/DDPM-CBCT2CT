import os
import glob
import numpy as np
import nibabel as nib
import pydicom
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from pydicom.uid import generate_uid, ExplicitVRLittleEndian
from pydicom.sequence import Sequence
import datetime


def nifti_to_dicom(nifti_path, output_dir, modality="CT", is_mask=False, series_description=""):
    os.makedirs(output_dir, exist_ok=True)

    nii = nib.load(nifti_path)
    volume = nii.get_fdata()

    if is_mask:
        volume = volume.astype(np.uint16)
        rescale_intercept = 0
    else:
        rescale_intercept = int(volume.min())
        volume = (volume - rescale_intercept).astype(np.uint16)

    H, W, n_slices = volume.shape
    zooms = nii.header.get_zooms()
    pixel_spacing    = [float(zooms[0]), float(zooms[1])]
    slice_thickness  = float(zooms[2]) if len(zooms) > 2 else 1.0

    series_instance_uid = generate_uid()
    study_instance_uid  = generate_uid()
    study_date = datetime.datetime.now().strftime("%Y%m%d")
    study_time = datetime.datetime.now().strftime("%H%M%S")

    for i in range(n_slices):
        slice_data = volume[:, :, i].astype(np.uint16)

        file_meta = FileMetaDataset()
        file_meta.MediaStorageSOPClassUID    = "1.2.840.10008.5.1.4.1.1.2"
        file_meta.MediaStorageSOPInstanceUID = generate_uid()
        file_meta.TransferSyntaxUID          = ExplicitVRLittleEndian
        file_meta.ImplementationClassUID     = generate_uid()  # ← tambahan

        ds = FileDataset(None, {}, file_meta=file_meta, preamble=b"\0" * 128)

        ds.PatientName      = "Anonymous"
        ds.PatientID        = os.path.basename(os.path.dirname(output_dir))
        ds.StudyInstanceUID = study_instance_uid
        ds.StudyDate        = study_date
        ds.StudyTime        = study_time
        ds.AccessionNumber  = ""

        ds.SeriesInstanceUID = series_instance_uid
        ds.SeriesNumber      = 1
        ds.Modality          = modality
        ds.SeriesDescription = series_description        # "cbct", "ct", atau "mask"
        ds.StudyDescription  = "Brain CBCT2CT"   # deskripsi bebas
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
        ds.PatientPosition         = "HFS"
        ds.SliceLocation           = float(i) * slice_thickness

        ds.SamplesPerPixel           = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.BitsAllocated             = 16
        ds.BitsStored                = 16
        ds.HighBit                   = 15
        ds.PixelRepresentation       = 0
        ds.RescaleSlope              = 1
        ds.RescaleIntercept          = rescale_intercept

        # ← cara yang benar untuk set pixel data di pydicom
        ds.PixelData = slice_data.tobytes()
        ds['PixelData'].VR = 'OB'   # ← fix utama

        out_path = os.path.join(output_dir, f"slice_{i+1:04d}.dcm")
        pydicom.dcmwrite(out_path, ds, write_like_original=False)

    print(f"  ✓ {n_slices} slices → {output_dir}")
    
def convert_patient(patient_dir, output_root):
    patient_id  = os.path.basename(patient_dir)
    patient_out = os.path.join(output_root, patient_id)

    modality_map = {
        "cbct.nii.gz": ("cbct", "CT", False),
        "ct.nii.gz":   ("ct",   "CT", False),
        "mask.nii.gz": ("mask", "CT", True),
    }

    print(f"\n[Patient ID] {patient_id}")
    for filename, (folder_name, modality, is_mask) in modality_map.items():
        nifti_path = os.path.join(patient_dir, filename)
        if not os.path.exists(nifti_path):
            print(f"  ✗ {filename} not found, skipping.")
            continue
        output_dir = os.path.join(patient_out, folder_name)
        nifti_to_dicom(nifti_path, output_dir, modality=modality, is_mask=is_mask, series_description=folder_name)


def convert_all(brain_root="Brain", output_root="Brain_DICOM"):
    patient_dirs = sorted(glob.glob(os.path.join(brain_root, "*")))
    if not patient_dirs:
        print(f"No subfolders found in '{brain_root}'")
        return

    print(f"Found {len(patient_dirs)} patients. Starting conversion...")
    for d in patient_dirs:
        if os.path.isdir(d):
            convert_patient(d, output_root)

    print(f"\nDone! Output saved at: {output_root}/")


if __name__ == "__main__":
    convert_all(
        brain_root  = "/data/3THDD/dataset/CBCT2CT/brain/",
        output_root = "/data/3THDD/dataset/CBCT2CT/brain_DICOM/"
    )
#    convert_patient(
#        patient_dir = "/data/3THDD/dataset/CBCT2CT/brain/2BB075/",
#        output_root = "/data/3THDD/dataset/CBCT2CT/brain_dicom_test/2BB075/"
#    )