import numpy as np
import SimpleITK as sitk
import pandas as pd
from utils import read_yaml
from pathlib import Path
from collections import OrderedDict
# Optional: convex hull features
try:
    from scipy.spatial import ConvexHull
    _HAS_SCIPY = True
    print("SciPy detected: convex hull features will be computed.")
except Exception:
    _HAS_SCIPY = False
    print("SciPy not detected: convex hull features will be skipped.")
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

def _binary_for_label(msk: sitk.Image, label: int) -> sitk.Image:
    """Return binary mask (UInt8) for one label."""
    return sitk.Cast(msk == int(label), sitk.sitkUInt8)

def _safe_quantiles(x: np.ndarray, qs=(0.1, 0.25, 0.5, 0.75, 0.9)):
    if x.size == 0:
        return {f"q{int(q*100):02d}": np.nan for q in qs}
    vals = np.quantile(x, qs)
    return {f"q{int(q*100):02d}": float(v) for q, v in zip(qs, vals)}

def _ball_radius_voxels(spacing_xyz, radius_mm: float):
    """Convert physical mm radius to voxel radii (x,y,z) for a ball structuring element."""
    sx, sy, sz = spacing_xyz
    rx = max(1, int(np.round(radius_mm / sx)))
    ry = max(1, int(np.round(radius_mm / sy)))
    rz = max(1, int(np.round(radius_mm / sz)))
    return [rx, ry, rz]

def _erode_mm(bin_mask: sitk.Image, radius_mm: float) -> sitk.Image:
    # Force strict binary mask (0/1) of type UInt8
    bin_mask = sitk.Cast(bin_mask > 0, sitk.sitkUInt8)

    # Convert mm -> voxels (x,y,z)
    r = _ball_radius_voxels(bin_mask.GetSpacing(), radius_mm)

    f = sitk.BinaryErodeImageFilter()
    f.SetKernelType(sitk.sitkBall)
    f.SetKernelRadius(r)
    f.SetForegroundValue(1)
    # Boundary handling: treat outside image as background (default is False, keep explicit)
    f.SetBoundaryToForeground(False)

    return f.Execute(bin_mask)



def _get_boundary_voxels(bin_mask: sitk.Image) -> sitk.Image:
    """1-voxel thick boundary mask."""
    return sitk.LabelContour(bin_mask)  # foreground=1, background=0

def _phys_coords_from_mask(mask_u8: sitk.Image, max_points: int = 20000) -> np.ndarray:
    """
    Extract physical coordinates of foreground voxels.
    To keep things fast, downsample if too many points.
    """
    arr = sitk.GetArrayFromImage(mask_u8)  # z,y,x
    idx = np.argwhere(arr > 0)
    if idx.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.float32)

    # downsample deterministically for speed if too many points
    if idx.shape[0] > max_points:
        step = int(np.ceil(idx.shape[0] / max_points))
        idx = idx[::step]

    # Convert (z,y,x) -> (x,y,z) index for TransformIndexToPhysicalPoint
    # We'll map indices in a loop (fast enough after downsampling)
    pts = np.zeros((idx.shape[0], 3), dtype=np.float32)
    for i, (z, y, x) in enumerate(idx):
        px, py, pz = mask_u8.TransformIndexToPhysicalPoint((int(x), int(y), int(z)))
        pts[i] = (px, py, pz)
    return pts





def compute_extra_shape_boundary_features(
    img: sitk.Image,
    msk: sitk.Image,
    label: int,
):

    """
    Extra non-PyRadiomics features from CT + binary mask for one label.

    - EDT thickness stats (all lesion voxels; includes boundary zeros)
    - Rim fractions (1mm, 2mm) using robust SITK boolean logic
    - Radial roughness (centroid -> boundary physical distances)
    - Gradient magnitude stats on 1mm rim; fallback to 1-voxel contour if rim empty
    - Convex hull (optional SciPy): hull volume/area, solidity, convexity (if surface area provided)
    """
    feats = OrderedDict()

    bin_mask = _binary_for_label(msk, label)

    # voxel count
    arr_bin = sitk.GetArrayFromImage(bin_mask).astype(np.uint8)
    vol_total_vox = int(arr_bin.sum())
    if vol_total_vox == 0:
        # should not happen if label exists, but keep safe
        for k in [
            "extra_edt_mean_mm","extra_edt_std_mm","extra_edt_max_mm",
            "extra_edt_q10_mm","extra_edt_q25_mm","extra_edt_q50_mm","extra_edt_q75_mm","extra_edt_q90_mm",
            "extra_rim1mm_frac","extra_rim2mm_frac",
            "extra_radial_std_mm","extra_radial_iqr_mm","extra_radial_p90_p10_mm","extra_radial_max_min_ratio",
            "extra_grad_rim1mm_mean","extra_grad_rim1mm_std","extra_grad_rim1mm_q10","extra_grad_rim1mm_q25",
            "extra_grad_rim1mm_q50","extra_grad_rim1mm_q75","extra_grad_rim1mm_q90",
            "extra_hull_volume_mm3","extra_hull_area_mm2","extra_solidity","extra_convexity",
        ]:
            feats[k] = np.nan
        return feats

    # --- EDT thickness ---
    dist = sitk.SignedMaurerDistanceMap(
        bin_mask, insideIsPositive=True, squaredDistance=False, useImageSpacing=True
    )
    dist_arr = sitk.GetArrayFromImage(dist)
    mask_arr = sitk.GetArrayFromImage(bin_mask).astype(bool)

    inside_all = dist_arr[mask_arr]                 # includes boundary zeros
    inside_pos = inside_all[inside_all > 0]         # interior-only

    # A) all voxels
    feats["extra_edt_mean_mm"] = float(np.mean(inside_all))
    feats["extra_edt_std_mm"]  = float(np.std(inside_all))
    feats["extra_edt_max_mm"]  = float(np.max(inside_all))
    q_all = _safe_quantiles(inside_all, qs=(0.1, 0.25, 0.5, 0.75, 0.9))
    feats["extra_edt_q10_mm"] = q_all["q10"]
    feats["extra_edt_q25_mm"] = q_all["q25"]
    feats["extra_edt_q50_mm"] = q_all["q50"]
    feats["extra_edt_q75_mm"] = q_all["q75"]
    feats["extra_edt_q90_mm"] = q_all["q90"]

    # B) interior-only (recommended to add)
    if inside_pos.size > 0:
        q_pos = _safe_quantiles(inside_pos, qs=(0.1, 0.25, 0.5, 0.75, 0.9))
        feats["extra_edt_interior_mean_mm"] = float(np.mean(inside_pos))
        feats["extra_edt_interior_std_mm"]  = float(np.std(inside_pos))
        feats["extra_edt_interior_max_mm"]  = float(np.max(inside_pos))
        feats["extra_edt_interior_q10_mm"]  = q_pos["q10"]
        feats["extra_edt_interior_q25_mm"]  = q_pos["q25"]
        feats["extra_edt_interior_q50_mm"]  = q_pos["q50"]
        feats["extra_edt_interior_q75_mm"]  = q_pos["q75"]
        feats["extra_edt_interior_q90_mm"]  = q_pos["q90"]
    else:
        feats["extra_edt_interior_mean_mm"] = np.nan
        feats["extra_edt_interior_std_mm"]  = np.nan
        feats["extra_edt_interior_max_mm"]  = np.nan
        feats["extra_edt_interior_q10_mm"]  = np.nan
        feats["extra_edt_interior_q25_mm"]  = np.nan
        feats["extra_edt_interior_q50_mm"]  = np.nan
        feats["extra_edt_interior_q75_mm"]  = np.nan
        feats["extra_edt_interior_q90_mm"]  = np.nan

    feats["extra_edt_zero_frac"] = float(np.mean(inside_all == 0))

    # --- Rim fractions (count-based, bulletproof) ---
    arr_m = sitk.GetArrayFromImage(bin_mask) > 0
    vol_total_vox = int(arr_m.sum())

    feats["extra_rim1mm_frac"] = np.nan
    feats["extra_rim2mm_frac"] = np.nan

    if vol_total_vox > 0:
        for rmm in (1.0, 2.0):
            er = _erode_mm(bin_mask, rmm)
            er_vox = int((sitk.GetArrayFromImage(er) > 0).sum())

            rim_vox = max(0, vol_total_vox - er_vox)  # erosion can zero out
            feats[f"extra_rim{int(rmm)}mm_frac"] = float(rim_vox / (vol_total_vox + 1e-8))



    # --- Radial roughness (centroid -> boundary distances) ---
    boundary = _get_boundary_voxels(bin_mask)
    pts = _phys_coords_from_mask(boundary, max_points=15000)

    if pts.shape[0] >= 10:
        stats = sitk.LabelShapeStatisticsImageFilter()
        stats.Execute(bin_mask)
        cx, cy, cz = stats.GetCentroid(1)

        r = np.sqrt((pts[:, 0] - cx) ** 2 + (pts[:, 1] - cy) ** 2 + (pts[:, 2] - cz) ** 2)
        feats["extra_radial_std_mm"] = float(np.std(r))
        feats["extra_radial_iqr_mm"] = float(np.quantile(r, 0.75) - np.quantile(r, 0.25))
        feats["extra_radial_p90_p10_mm"] = float(np.quantile(r, 0.90) - np.quantile(r, 0.10))
        feats["extra_radial_max_min_ratio"] = float(np.max(r) / (np.min(r) + 1e-8))
    else:
        feats["extra_radial_std_mm"] = np.nan
        feats["extra_radial_iqr_mm"] = np.nan
        feats["extra_radial_p90_p10_mm"] = np.nan
        feats["extra_radial_max_min_ratio"] = np.nan

    # --- Gradient magnitude stats on 1mm rim; fallback to contour if rim empty ---
    grad = sitk.GradientMagnitude(img)
    grad_arr = sitk.GetArrayFromImage(grad)

    er1 = _erode_mm(bin_mask, 1.0)
    rim1_img = sitk.And(bin_mask, sitk.BinaryNot(er1))
    rim1 = sitk.GetArrayFromImage(rim1_img).astype(bool)

    if not np.any(rim1):
        # fallback to 1-voxel boundary
        contour_img = _get_boundary_voxels(bin_mask)
        rim1 = sitk.GetArrayFromImage(contour_img).astype(bool)

    if np.any(rim1):
        g = grad_arr[rim1]
        feats["extra_grad_rim1mm_mean"] = float(np.mean(g))
        feats["extra_grad_rim1mm_std"]  = float(np.std(g))
        qg = _safe_quantiles(g, qs=(0.1, 0.25, 0.5, 0.75, 0.9))
        feats["extra_grad_rim1mm_q10"] = qg["q10"]
        feats["extra_grad_rim1mm_q25"] = qg["q25"]
        feats["extra_grad_rim1mm_q50"] = qg["q50"]
        feats["extra_grad_rim1mm_q75"] = qg["q75"]
        feats["extra_grad_rim1mm_q90"] = qg["q90"]
    else:
        feats["extra_grad_rim1mm_mean"] = np.nan
        feats["extra_grad_rim1mm_std"]  = np.nan
        feats["extra_grad_rim1mm_q10"] = np.nan
        feats["extra_grad_rim1mm_q25"] = np.nan
        feats["extra_grad_rim1mm_q50"] = np.nan
        feats["extra_grad_rim1mm_q75"] = np.nan
        feats["extra_grad_rim1mm_q90"] = np.nan

    # --- Optional: convex hull features (SciPy) ---
    feats["extra_hull_volume_mm3"] = np.nan
    feats["extra_hull_area_mm2"]   = np.nan
    feats["extra_solidity"]        = np.nan
    # feats["extra_convexity"]       = np.nan

    if _HAS_SCIPY and pts.shape[0] >= 20:
        try:
            hull = ConvexHull(pts)  # pts are physical => volume in mm^3, area in mm^2
            Vh = float(hull.volume)
            Sh = float(hull.area)
            feats["extra_hull_volume_mm3"] = Vh
            feats["extra_hull_area_mm2"]   = Sh

            sx, sy, sz = msk.GetSpacing()
            V = float(vol_total_vox * sx * sy * sz)
            feats["extra_solidity"] = float(V / (Vh + 1e-8))

        #     if surface_area_from_radiomics is not None and np.isfinite(surface_area_from_radiomics):
        #         feats["extra_convexity"] = float(Sh / (float(surface_area_from_radiomics) + 1e-8))
        except Exception:
            pass
    
    return feats




def same_geometry(img, msk) -> bool:
    return (
        img.GetSize() == msk.GetSize()
        and np.allclose(img.GetSpacing(), msk.GetSpacing())
        and np.allclose(img.GetOrigin(), msk.GetOrigin())
        and np.allclose(img.GetDirection(), msk.GetDirection())
    )

def resample_mask_to_image(mask, ref_img):
    """Nearest-neighbor resample mask onto ref_img grid."""
    res = sitk.ResampleImageFilter()
    res.SetReferenceImage(ref_img)
    res.SetInterpolator(sitk.sitkNearestNeighbor)
    res.SetTransform(sitk.Transform())
    res.SetDefaultPixelValue(0)
    return res.Execute(mask)


def extract_lesion_shape_features(
    ct_path: str,
    mask_path: str,
    min_voxels: int = 50,
    out_csv_path: str = None,
):
    img = sitk.ReadImage(ct_path)
    msk = sitk.ReadImage(mask_path)
    # check if mask has values
    arr_mask = sitk.GetArrayFromImage(msk)
    if np.sum(arr_mask) == 0:
        return 
    # Ensure mask is on image grid
    if not same_geometry(img, msk):
        msk = resample_mask_to_image(msk, img)

    arr = sitk.GetArrayFromImage(msk)
    labels = [int(x) for x in np.unique(arr) if int(x) != 0]
    if len(labels) == 0:
        return
        # raise ValueError("No lesion labels found in mask (all zeros).")

    lesion_rows = []
    for lab in labels:
        nvox = int(np.sum(arr == lab))
        if nvox < min_voxels:
            continue

        # Execute radiomics for this label
        feats = compute_extra_shape_boundary_features(img, msk, label=lab)

        # Add metadata
        feats["lesion_label"] = lab
        lesion_rows.append(feats)

    if len(lesion_rows) == 0:
        raise ValueError(f"All lesions were below min_voxels={min_voxels}.")

    df_lesions = pd.DataFrame(lesion_rows)
    # Replace NaN with -1
    df_lesions = df_lesions.fillna(-1.0)
    df_lesions.to_csv(out_csv_path, index=False)


def perform_one_extraction(args):
    ct_path, mask_path, min_voxels, out_csv_path = args
    extract_lesion_shape_features(
                str(ct_path),
                str(mask_path),
                min_voxels=min_voxels,
                out_csv_path=out_csv_path,
            )


def main(data_config_dir):
    data_config = read_yaml(data_config_dir)

    preprocessed_data_base_dir = data_config["preprocessed_data_base_dir"]
    ct_base_dir = Path(preprocessed_data_base_dir) / "04_images_resampled_marginal_cropped"
    seg_base_dir = Path(preprocessed_data_base_dir) / "04_segmentations_resampled_marginal_cropped"

    output_dir = Path(preprocessed_data_base_dir) / "06_extra_shape_features"
    output_dir.mkdir(parents=True, exist_ok=True)

    seg_paths = sorted(list(seg_base_dir.rglob("*.nii.gz")))
    img_paths = sorted(list(ct_base_dir.rglob("*.nii")))

    # sanity check
    print(f"Found {len(img_paths)} images and {len(seg_paths)} segmentations.")
    assert len(img_paths) == len(seg_paths), "Number of images and segmentations do not match."
    print(seg_paths[0].name, img_paths[0].name)

    tasks = []
    for ct_path, seg_path in zip(img_paths, seg_paths):
        img_id = ct_path.name
        out_csv_path = output_dir / f"{img_id.replace('.nii', '_extra_shape_features.csv')}"
        tasks.append((str(ct_path), str(seg_path), data_config.get("radiomics_min_voxels", 50), str(out_csv_path)))
        
        # try
        # extract_lesion_shape_features(
        #     str(ct_path),
        #     str(seg_path),
        #     min_voxels=data_config.get("radiomics_min_voxels", 50),
        #     out_csv_path=str(out_csv_path)
        # )
        # exit()
    
    with ProcessPoolExecutor(max_workers=10) as executor:
        list(tqdm(executor.map(perform_one_extraction, tasks), total=len(tasks)))
        
if __name__ == "__main__":
    data_config_dir = '../configs/data_config.yaml'
    main(data_config_dir)