"""
npy_to_dicom.py
-----------------------------------------------------------------------------
Konversi sCT (.npy, HU float32) -> DICOM series per pasien.

Kalibrasi HU (cegah bug intercept):
    RescaleIntercept=-1024, RescaleSlope=1, RescaleType='HU'
    stored(int16) = round(HU + 1024)  ->  viewer baca HU identik dgn .npy

Geometri / bentuk (oval vs bulat):
    Data dilatih di 256x256, sementara CBCT/CT asli landscape (mis 289x245,
    FOV W/H ~1.18). Resize-ke-persegi membuat sCT tampak lebih bulat.
    GRID_MODE mengatur cara mengembalikan bentuk yang benar:

      "match_ct"     : RESAMPLE sCT 256x256 -> grid asli CT pasien (mis 245x289),
                       PixelSpacing ikut CT. sCT jadi SEGRID dgn CT/GT (bisa
                       overlay). Catatan: resize sedikit menghaluskan HU tepi.
      "display_only" : TIDAK resample. Tetap 256x256, tapi PixelSpacing dibuat
                       anisotropik supaya FOV fisik = native -> bentuk oval pulih
                       di viewer. HU bit-exact (tidak disentuh).
      "square"       : perilaku lama, 256x256 + spacing isotropik (PIXEL_SPACING).

Grid asli dibaca otomatis dari REF_DICOM_ROOT/<patient>/REF_SUB/.
-----------------------------------------------------------------------------
"""

import os
import re
import glob
import numpy as np
import datetime

import pydicom
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import generate_uid, ExplicitVRLittleEndian
from skimage.transform import resize as sk_resize

CT_IMAGE_STORAGE = "1.2.840.10008.5.1.4.1.1.2"

# ============================== CONFIG ======================================
NPY_ROOT = "./test/test/npy"
OUT_ROOT = "./test/test/dicom_sCT_post_process"

# index sCT kalau file .npy berbentuk (3,H,W)=[CBCT,Fake,GT]; diabaikan jika (H,W)
SCT_INDEX = 1

# --- kalibrasi HU (JANGAN diubah tanpa alasan) ---
RESCALE_INTERCEPT = -1024
RESCALE_SLOPE = 1

# --- geometri / bentuk ---
GRID_MODE = "match_ct"      # "match_ct" | "display_only" | "square"
REF_DICOM_ROOT = "/data/3THDD/dataset/CBCT2CT/brain_DICOM"
REF_SUB = "ct"              # baca grid asli dari subfolder ini (ct atau cbct, sama)

# dipakai HANYA kalau GRID_MODE="square" atau geometri referensi tak ditemukan
PIXEL_SPACING = [1.0, 1.0]  # [row, col] mm
SLICE_THICKNESS = 1.0       # mm
SLICE_SPACING = 1.0         # mm, increment z antar slice

# --- window tampilan (HANYA preset; HU tidak berubah) ---
REF_CT_DICOM = None         # path 1 file CT utk salin window; None -> preset bawah
WINDOW_CENTER = [40, 500]   # [brain, bone]
WINDOW_WIDTH = [80, 2000]   # [brain, bone]

HU_CLIP = (-1024, 3071)     # safety net agar muat int16; None = tidak clip
# ===========================================================================


def natural_key(path):
    name = os.path.splitext(os.path.basename(path))[0]
    nums = re.findall(r"\d+", name)
    return (int(nums[0]) if nums else 0, name)


def list_patients(root):
    return [d for d in sorted(os.listdir(root)) if os.path.isdir(os.path.join(root, d))]


def find_slice_files(patient_dir):
    sct_dir = os.path.join(patient_dir, "sCT")
    src_dir = sct_dir if os.path.isdir(sct_dir) else patient_dir
    files = glob.glob(os.path.join(src_dir, "*.npy"))
    files = [f for f in files if "mask" not in os.path.basename(f).lower()]
    return sorted(files, key=natural_key)


def load_sct_hu(path):
    arr = np.asarray(np.load(path))
    if arr.ndim == 3:
        if arr.shape[0] == 1:
            arr = arr[0]
        elif arr.shape[0] > SCT_INDEX:
            arr = arr[SCT_INDEX]
        else:
            raise ValueError(f"Bentuk tak terduga {arr.shape} di {path}")
    if arr.ndim != 2:
        raise ValueError(f"Slice bukan 2D: {arr.shape} di {path}")
    return arr.astype(np.float32)


def hu_to_stored(hu):
    if HU_CLIP is not None:
        hu = np.clip(hu, HU_CLIP[0], HU_CLIP[1])
    stored = np.rint((hu - RESCALE_INTERCEPT) / RESCALE_SLOPE)
    return np.clip(stored, -32768, 32767).astype(np.int16)


def resolve_window():
    if REF_CT_DICOM:
        ref = pydicom.dcmread(REF_CT_DICOM, stop_before_pixels=True)
        return ref.get("WindowCenter", WINDOW_CENTER), ref.get("WindowWidth", WINDOW_WIDTH)
    return WINDOW_CENTER, WINDOW_WIDTH


def resolve_patient_geometry(patient):
    """Baca grid asli (rows, cols, row_sp, col_sp) dari DICOM CT pasien. None jika gagal."""
    ref_dir = os.path.join(REF_DICOM_ROOT, patient, REF_SUB)
    files = sorted(glob.glob(os.path.join(ref_dir, "*.dcm")))
    if not files:
        return None
    d = pydicom.dcmread(files[len(files) // 2], stop_before_pixels=True)
    rows = int(getattr(d, "Rows", 0))
    cols = int(getattr(d, "Columns", 0))
    ps = d.get("PixelSpacing", [1.0, 1.0])
    if rows == 0 or cols == 0:
        return None
    return rows, cols, float(ps[0]), float(ps[1])


def make_dicom_slice(stored2d, patient_id, study_uid, series_uid, frame_uid,
                     series_number, instance_number, z_pos,
                     pixel_spacing, win_center, win_width):
    rows, cols = stored2d.shape
    now = datetime.datetime.now()
    sop_uid = generate_uid()

    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = CT_IMAGE_STORAGE
    file_meta.MediaStorageSOPInstanceUID = sop_uid
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    file_meta.ImplementationClassUID = generate_uid()

    ds = FileDataset(None, {}, file_meta=file_meta, preamble=b"\x00" * 128)
    ds.is_little_endian = True
    ds.is_implicit_VR = False

    ds.PatientName = str(patient_id)
    ds.PatientID = str(patient_id)
    ds.PatientBirthDate = ""
    ds.PatientSex = ""
    ds.PatientPosition = "HFS"

    ds.StudyInstanceUID = study_uid
    ds.StudyDate = now.strftime("%Y%m%d")
    ds.StudyTime = now.strftime("%H%M%S")
    ds.StudyID = "1"
    ds.AccessionNumber = ""
    ds.ReferringPhysicianName = ""

    ds.SeriesInstanceUID = series_uid
    ds.SeriesNumber = series_number
    ds.Modality = "CT"
    ds.SeriesDescription = "synthetic CT (sCT)"

    ds.FrameOfReferenceUID = frame_uid
    ds.PositionReferenceIndicator = ""

    ds.SOPClassUID = CT_IMAGE_STORAGE
    ds.SOPInstanceUID = sop_uid

    ds.InstanceNumber = instance_number
    ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
    ds.ImagePositionPatient = [0.0, 0.0, float(z_pos)]
    ds.SliceLocation = float(z_pos)
    ds.PixelSpacing = [float(pixel_spacing[0]), float(pixel_spacing[1])]
    ds.SliceThickness = float(SLICE_THICKNESS)
    ds.SpacingBetweenSlices = float(SLICE_SPACING)

    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.Rows = rows
    ds.Columns = cols
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 1

    ds.RescaleIntercept = RESCALE_INTERCEPT
    ds.RescaleSlope = RESCALE_SLOPE
    ds.RescaleType = "HU"

    ds.WindowCenter = win_center
    ds.WindowWidth = win_width

    ds.PixelData = stored2d.tobytes()
    return ds


def plan_geometry(patient, src_shape):
    """Tentukan (target_shape, pixel_spacing, do_resize) sesuai GRID_MODE."""
    src_rows, src_cols = src_shape
    geo = resolve_patient_geometry(patient) if GRID_MODE in ("match_ct", "display_only") else None

    if GRID_MODE == "match_ct" and geo:
        nr, nc, rsp, csp = geo
        return (nr, nc), [rsp, csp], True, True
    if GRID_MODE == "display_only" and geo:
        nr, nc, rsp, csp = geo
        phys_h, phys_w = nr * rsp, nc * csp
        return (src_rows, src_cols), [phys_h / src_rows, phys_w / src_cols], False, True
    # fallback / square
    fell_back = (GRID_MODE in ("match_ct", "display_only"))  # geo gagal
    return (src_rows, src_cols), list(PIXEL_SPACING), False, (not fell_back)


def convert_patient(patient, in_root, out_root, win_center, win_width):
    files = find_slice_files(os.path.join(in_root, patient))
    if not files:
        return 0, True

    first = load_sct_hu(files[0])
    target, pixel_spacing, do_resize, geo_ok = plan_geometry(patient, first.shape)

    out_dir = os.path.join(out_root, patient)
    os.makedirs(out_dir, exist_ok=True)
    study_uid, series_uid, frame_uid = generate_uid(), generate_uid(), generate_uid()

    for i, f in enumerate(files):
        hu = first if i == 0 else load_sct_hu(f)
        if do_resize and hu.shape != target:
            hu = sk_resize(hu, target, order=1, preserve_range=True,
                           anti_aliasing=True).astype(np.float32)
        stored = hu_to_stored(hu)
        ds = make_dicom_slice(
            stored, patient, study_uid, series_uid, frame_uid,
            series_number=1, instance_number=i + 1, z_pos=i * SLICE_SPACING,
            pixel_spacing=pixel_spacing, win_center=win_center, win_width=win_width,
        )
        ds.save_as(os.path.join(out_dir, f"{i + 1:04d}.dcm"), write_like_original=False)

    return len(files), geo_ok


def main():
    os.makedirs(OUT_ROOT, exist_ok=True)
    win_center, win_width = resolve_window()
    patients = list_patients(NPY_ROOT)

    total, summary, warn = 0, [], []
    for p in patients:
        n, geo_ok = convert_patient(p, NPY_ROOT, OUT_ROOT, win_center, win_width)
        if n > 0:
            summary.append((p, n))
            total += n
            if not geo_ok:
                warn.append(p)

    print(f"Selesai. GRID_MODE={GRID_MODE}. {len(summary)} pasien, {total} slice -> {OUT_ROOT}")
    for p, n in summary:
        print(f"  {p}: {n} slice")
    if warn:
        print("\nPERINGATAN: geometri referensi tak ditemukan, pakai PIXEL_SPACING default utk:")
        for p in warn:
            print(f"  {p}")


if __name__ == "__main__":
    main()