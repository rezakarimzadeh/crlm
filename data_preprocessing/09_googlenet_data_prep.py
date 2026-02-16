import numpy as np
import SimpleITK as sitk
import pandas as pd
from radiomics import featureextractor
from utils import read_yaml
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from PIL import Image

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



def clip_hu_for_radiomics(img: sitk.Image, hu_min: float = -200.0, hu_max: float = 300.0) -> sitk.Image:
    """
    Clip intensities to [hu_min, hu_max] without rescaling.
    Keeps values in HU-like units (important if you insist on windowing for radiomics).
    Output type matches input type.
    """
    in_id = img.GetPixelID()
    img_f = sitk.Cast(img, sitk.sitkFloat32)

    clipped = sitk.Clamp(img_f, lowerBound=float(hu_min), upperBound=float(hu_max))

    return sitk.Cast(clipped, in_id)



def array_to_sitk_image(array, reference_sitk_image):
    sitk_image = sitk.GetImageFromArray(array)
    sitk_image.CopyInformation(reference_sitk_image)
    return sitk_image


def extent_mm_along_axes(mask_zyx, spacing_xyz):
    sz, sy, sx = spacing_xyz[2], spacing_xyz[1], spacing_xyz[0]  # map to z,y,x
    coords = np.argwhere(mask_zyx > 0)
    zmin, ymin, xmin = coords.min(axis=0)
    zmax, ymax, xmax = coords.max(axis=0)
    # dz = (zmax - zmin + 1) * sz
    dy = (ymax - ymin + 1) * sy
    dx = (xmax - xmin + 1) * sx
    max_diameter_mm = max(dx, dy)  # only consider in-plane diameter for 2D slices
    return max_diameter_mm


def crop_2diameter_center_of_mass(img_array, mask_zyx, spacing_xyz, diameter_mm):
    sz, sy, sx = spacing_xyz[2], spacing_xyz[1], spacing_xyz[0]  # map to z,y,x
    radius_mm = diameter_mm / 2
    radius_voxels_y = int(np.ceil(radius_mm / sy))
    radius_voxels_x = int(np.ceil(radius_mm / sx))

    center_of_mass = np.mean(np.argwhere(mask_zyx > 0), axis=0).astype(int)  # in z,y,x order
    zc, yc, xc = center_of_mass

    y_min = max(0, yc - radius_voxels_y)
    y_max = min(mask_zyx.shape[1], yc + radius_voxels_y + 1)
    x_min = max(0, xc - radius_voxels_x)
    x_max = min(mask_zyx.shape[2], xc + radius_voxels_x + 1)

    cropped_img = img_array[zc, y_min:y_max, x_min:x_max]
    return cropped_img


def normalize_to_0_255(cropped_img):
    # Normalize to [0, 255] for 8-bit representation
    min_val = cropped_img.min()
    max_val = cropped_img.max()
    if max_val > min_val:  # avoid division by zero
        norm_img = (cropped_img - min_val) / (max_val - min_val) * 255.0
    else:
        norm_img = np.zeros_like(cropped_img)
    return norm_img.astype(np.uint8)


def resize_and_save(img_2d, output_path, new_size):
    pil_img = Image.fromarray(img_2d)
    pil_img = pil_img.resize(new_size, Image.BILINEAR)
    pil_img.save(output_path)


def extract_2d_images_of_tumors(
    ct_path: str,
    mask_path: str,
    out_path: str ,
):
    img = sitk.ReadImage(ct_path)
    msk = sitk.ReadImage(mask_path)

    # Ensure mask is on image grid first
    if not same_geometry(img, msk):
        msk = resample_mask_to_image(msk, img)


    img = clip_hu_for_radiomics(img, hu_min=-150, hu_max=250)
    # img = clip_hu_for_radiomics(img, hu_min=-135, hu_max=215)
    img_array = sitk.GetArrayFromImage(img)

    arr = sitk.GetArrayFromImage(msk)
    if not np.any(arr > 0):
        return

    labels = [int(x) for x in np.unique(arr) if int(x) != 0]

    for lab in labels:
        if lab == 1:
            continue  # skip liver

        label_mask = (arr == lab).astype(np.uint8)
        max_diameter_mm = extent_mm_along_axes(label_mask, img.GetSpacing())
        if max_diameter_mm < 10:  # skip very small lesions < 1cm
            continue
        
        cropped_img = crop_2diameter_center_of_mass(img_array, label_mask, img.GetSpacing(), diameter_mm=2*max_diameter_mm)
        cropped_img = normalize_to_0_255(cropped_img)

        out_file = f"{out_path}/diameter_{int(max_diameter_mm)}_label_{lab}.png"
        out_file = Path(out_file)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        
        resize_and_save(cropped_img, out_file, new_size=(224, 224))

        

        


def perform_one_extraction(args):
    ct_path, mask_path, bin_width, min_voxels, out_path = args
    df_lesions = extract_2d_images_of_tumors(
                str(ct_path),
                str(mask_path),
                out_path=out_path,
            )


def main(data_config_dir):
    data_config = read_yaml(data_config_dir)

    preprocessed_data_base_dir = data_config["preprocessed_data_base_dir"]
    ct_base_dir = Path(preprocessed_data_base_dir) / "04_images_resampled_marginal_cropped"
    seg_base_dir = Path(preprocessed_data_base_dir) / "04_segmentations_resampled_marginal_cropped"

    output_dir = Path(preprocessed_data_base_dir) / "09_googlenet_2d_slices"
    output_dir.mkdir(parents=True, exist_ok=True)

    seg_paths = sorted(list(seg_base_dir.rglob("*.nii.gz")))
    img_paths = sorted(list(ct_base_dir.rglob("*.nii")))
    # filter imgs based on seg names
    # seg_ids = list([p.name.split(".nii.gz")[0] for p in seg_paths])
    # img_paths = [Path(ct_base_dir) / f"{id}_0000.nii.gz" for id in seg_ids]
    # sanity check
    print(f"Found {len(img_paths)} images and {len(seg_paths)} segmentations.")
    assert len(img_paths) == len(seg_paths), "Number of images and segmentations do not match."
    print(seg_paths[0].name, img_paths[0].name)

    tasks = []
    for ct_path, seg_path in zip(img_paths, seg_paths):
        img_id = ct_path.name
        out_path = output_dir / f"{img_id.replace('.nii', '')}"
        tasks.append((str(ct_path), str(seg_path), data_config.get("radiomics_bin_width", 15), data_config.get("radiomics_min_voxels", 50), str(out_path)))

    # perform_one_extraction(tasks[10])  # test on first case
    
    with ProcessPoolExecutor(max_workers=10) as executor:
        list(tqdm(executor.map(perform_one_extraction, tasks), total=len(tasks)))


if __name__ == "__main__":
    data_config_dir = '../configs/data_config.yaml'
    main(data_config_dir)