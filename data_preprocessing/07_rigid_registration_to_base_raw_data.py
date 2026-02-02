import argparse
import shutil
import SimpleITK as sitk
from utils import read_yaml, read_json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

def _same_geometry(a: sitk.Image, b: sitk.Image) -> bool:
    return (
        a.GetSize() == b.GetSize()
        and a.GetSpacing() == b.GetSpacing()
        and a.GetOrigin() == b.GetOrigin()
        and a.GetDirection() == b.GetDirection()
    )

def _resample_to_reference(moving: sitk.Image, reference: sitk.Image, is_label: bool) -> sitk.Image:
    interp = sitk.sitkNearestNeighbor if is_label else sitk.sitkLinear
    return sitk.Resample(
        moving,
        reference,          # reference grid
        sitk.Transform(),   # identity
        interp,
        0,                  # default value outside
        moving.GetPixelID()
    )

def maybe_apply_mask(ct: sitk.Image, mask: sitk.Image) -> sitk.Image:
    if mask is None:
        return ct

    # Convert any non-zero label to 1
    # Ensure scalar + predictable type
    mask_u16 = sitk.Cast(mask, sitk.sitkUInt16)

    mm = sitk.MinimumMaximumImageFilter()
    mm.Execute(mask_u16)
    mx = int(mm.GetMaximum())

    # if mask is empty (all zeros), skip masking
    if mx < 1:
        return ct

    mask_bin = sitk.BinaryThreshold(mask_u16, 1, mx, 1, 0)


    # Ensure the mask lives on the CT grid
    if not _same_geometry(ct, mask_bin):
        mask_bin = _resample_to_reference(mask_bin, ct, is_label=True)

    # Apply mask robustly (outsideValue can be -1024 if you prefer)
    return sitk.Mask(ct, mask_bin, outsideValue=0)

def ensure_valid_geometry(img: sitk.Image, name: str = "img") -> sitk.Image:
    sp = list(img.GetSpacing())
    if any(s <= 0 for s in sp):
        raise ValueError(f"{name} has invalid spacing: {sp} | size={img.GetSize()}")

    # also guard against NaNs
    if any((s != s) for s in sp):  # NaN check
        raise ValueError(f"{name} has NaN spacing: {sp} | size={img.GetSize()}")
    return img

def read_img(path):
    img = sitk.ReadImage(path)
    return sitk.Cast(img, sitk.sitkFloat32)


def read_seg(path):
    seg = sitk.ReadImage(path)
    # keep label type (often UInt8/UInt16). We'll cast to UInt16 for safety.
    return sitk.Cast(seg, sitk.sitkUInt16)


def correct_orientation(img, target_orientation="RAS"):
    return sitk.DICOMOrient(img, target_orientation)


def rigid_register(fixed_ct, moving_ct, fixed_mask=None, moving_mask=None,
                   sampling_pct=0.2, n_iter=200):
    """
    Rigid (Euler3D) registration using Mattes Mutual Information.
    Returns final transform.
    """
    # Optional ROI masking (common for follow-up abdominal CT)
    fixed_for_reg = maybe_apply_mask(fixed_ct, fixed_mask)
    moving_for_reg = maybe_apply_mask(moving_ct, moving_mask)

    # Initial alignment: center-of-geometry
    init_tx = sitk.CenteredTransformInitializer(
        fixed_for_reg, moving_for_reg,
        sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY
    )

    reg = sitk.ImageRegistrationMethod()
    reg.SetInitialTransform(init_tx, inPlace=False)

    # Similarity metric (robust default for CT follow-ups / contrast changes)
    reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    reg.SetMetricSamplingStrategy(reg.RANDOM)
    reg.SetMetricSamplingPercentage(sampling_pct, seed=42)
    reg.SetInterpolator(sitk.sitkLinear)

    reg.SetOptimizerAsRegularStepGradientDescent(
        learningRate=2.0,
        minStep=1e-4,
        numberOfIterations=n_iter,
        gradientMagnitudeTolerance=1e-8
    )
    reg.SetOptimizerScalesFromPhysicalShift()

    # Multi-resolution pyramid
    reg.SetShrinkFactorsPerLevel([4, 2, 1])
    reg.SetSmoothingSigmasPerLevel([2, 1, 0])
    reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    fixed_for_reg  = ensure_valid_geometry(fixed_for_reg,  "fixed_for_reg")
    moving_for_reg = ensure_valid_geometry(moving_for_reg, "moving_for_reg")

    final_tx = reg.Execute(fixed_for_reg, moving_for_reg)
    return final_tx


def resample_to_fixed(moving, fixed, transform, is_label=False, default_value=0):
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixed)
    resampler.SetTransform(transform)
    resampler.SetDefaultPixelValue(default_value)

    if is_label:
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
        # preserve label type
        return sitk.Cast(resampler.Execute(moving), sitk.sitkUInt16)
    else:
        resampler.SetInterpolator(sitk.sitkLinear)
        return sitk.Cast(resampler.Execute(moving), sitk.sitkFloat32)



def perform_one_case(args, sampling_pct=0.02, n_iter=200):
    fix_ct_path, moving_ct_path, fix_seg_path, moving_seg_path, img_registered_output_path, seg_registered_output_path = args
    # check if output already exists
    # if img_registered_output_path.exists() and seg_registered_output_path.exists():
    #     print(f"Skipping {img_registered_output_path.name} as output already exists.")
    #     return
    
    # ============================
    fixed_ct = read_img(fix_ct_path)
    moving_ct = read_img(moving_ct_path)
    moving_seg = read_seg(moving_seg_path)

    fixed_mask = read_seg(fix_seg_path) if fix_seg_path else None
    
    # merge all labels >0 into binary mask
    # default: use the moving segmentation as ROI (often tumor mask; for ROI masking
    # better to pass a liver/abdomen mask instead)
    moving_mask = sitk.Cast(moving_seg > 0, sitk.sitkUInt8)

    try:
        tx = rigid_register(
            fixed_ct=fixed_ct,
            moving_ct=moving_ct,
            fixed_mask=fixed_mask,
            moving_mask=moving_mask,
            sampling_pct=sampling_pct,
            n_iter=n_iter
        )

        moving_ct_reg = resample_to_fixed(moving_ct, fixed_ct, tx, is_label=False, default_value=-1024)
        moving_seg_reg = resample_to_fixed(moving_seg, fixed_ct, tx, is_label=True, default_value=0)

        sitk.WriteImage(moving_ct_reg, img_registered_output_path)
        sitk.WriteImage(moving_seg_reg, seg_registered_output_path)
        # sitk.WriteTransform(tx, out_transform)
    except Exception as e:
        print(f"Error processing {fix_ct_path}: {e}")

def main(data_config_dir):
    data_config = read_yaml(data_config_dir)
    preprocessed_data_base_dir =Path(data_config["preprocessed_data_base_dir"])
    raw_images_dir = preprocessed_data_base_dir / "04_images_resampled_marginal_cropped"
    liver_nodes_instances_dir =preprocessed_data_base_dir / "04_segmentations_resampled_marginal_cropped"

    output_images_dir = preprocessed_data_base_dir / "07_images_rigid_registration_cropped_to_base"
    output_segmentations_dir = preprocessed_data_base_dir / "07_segmentations_rigid_registration_cropped_to_base"
    output_segmentations_dir.mkdir(parents=True, exist_ok=True)
    output_images_dir.mkdir(parents=True, exist_ok=True)

    img_groups_dir = Path(preprocessed_data_base_dir) / "img_groups.json"
    img_groups = read_json(img_groups_dir)
    tasks = []
    for group in tqdm(img_groups):
        fix_ct_path = Path(raw_images_dir) / group["base_img"]
        fix_seg_path = Path(liver_nodes_instances_dir) / group["base_seg"]
        moving_ct_paths = [Path(raw_images_dir) / img_id for img_id in group["follow_ups_imgs"]]
        moving_seg_paths = [Path(liver_nodes_instances_dir) / seg_id for seg_id in group["follow_ups_segs"]]

        # check that paths exist
        if not fix_ct_path.exists() or not fix_seg_path.exists():
            print(f"Skipping group with base image {fix_ct_path.name} as base image/segmentation not found.")
            continue
        # copy base image/seg as well
        # check if already exists
        if not (output_images_dir / fix_ct_path.name).exists() and  not (output_segmentations_dir / fix_seg_path.name).exists():
            shutil.copyfile(fix_ct_path, output_images_dir / fix_ct_path.name)
            shutil.copyfile(fix_seg_path, output_segmentations_dir / fix_seg_path.name)
            print(f"Copied base image and segmentation for {fix_ct_path.name}.")

        # define tasks for follow-ups
        for moving_ct_path, moving_seg_path in zip(moving_ct_paths, moving_seg_paths):
            # check that paths exist
            if not moving_ct_path.exists() or not moving_seg_path.exists():
                print(f"Skipping follow-up image {moving_ct_path.name} as image/segmentation not found.")
                continue
            img_registered_output_path = output_images_dir / moving_ct_path.name
            seg_registered_output_path = output_segmentations_dir / moving_seg_path.name
            tasks.append((str(fix_ct_path), str(moving_ct_path), str(fix_seg_path), str(moving_seg_path),
                          str(img_registered_output_path), str(seg_registered_output_path)))


            # perform_one_case(tasks[0])  # for debugging
            # exit()

    with ProcessPoolExecutor(max_workers=10) as executor:
        list(tqdm(executor.map(perform_one_case, tasks), total=len(tasks)))




if __name__ == "__main__":
    data_config_dir = '../configs/data_config.yaml'
    main(data_config_dir)
