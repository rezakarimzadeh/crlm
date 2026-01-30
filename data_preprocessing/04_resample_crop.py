import SimpleITK as sitk
import numpy as np
from pathlib import Path
from utils import read_json, read_3d_volume, read_yaml
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

def correct_orientation(img, target_orientation="RAS"):
    return sitk.DICOMOrient(img, target_orientation)



def resample_img_seg(img, new_spacing, is_label=False):
    original_size = img.GetSize()        # (x,y,z)
    original_spacing = img.GetSpacing()  # (x,y,z)

    new_size = [
        int(np.round(original_size[i] * (original_spacing[i] / new_spacing[i])))
        for i in range(3)
    ]

    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(new_spacing)
    resample.SetSize(new_size)
    resample.SetOutputDirection(img.GetDirection())
    resample.SetOutputOrigin(img.GetOrigin())
    resample.SetTransform(sitk.Transform())  # identity

    if is_label:
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resample.SetInterpolator(sitk.sitkBSpline) 

    return resample.Execute(img)



def crop_img_seg(img, seg, crop_dimensions_mm):
    # find the center of the segmentation
    seg_array = sitk.GetArrayFromImage(seg)
    z_indices, y_indices, x_indices = seg_array.nonzero()
    z_center = int((z_indices.min() + z_indices.max()) / 2)
    y_center = int((y_indices.min() + y_indices.max()) / 2)
    x_center = int((x_indices.min() + x_indices.max()) / 2)
    center_index = [x_center, y_center, z_center]  # Note the order: x, y, z
    # calculate the crop size in voxels
    spacing = img.GetSpacing()
    crop_size_voxels = [
        int(crop_dimensions_mm[i] / spacing[i])
        for i in range(3)
    ]
    # calculate the start and end indices for cropping
    start_index = [
        max(0, center_index[i] - crop_size_voxels[i] // 2)
        for i in range(3)
    ]
    end_index = [
        min(img.GetSize()[i], start_index[i] + crop_size_voxels[i])
        for i in range(3)
    ]
    # adjust start index if end index is at the boundary
    for i in range(3):
        if end_index[i] - start_index[i] < crop_size_voxels[i]:
            start_index[i] = max(0, end_index[i] - crop_size_voxels[i])
    # perform cropping
    roi_filter = sitk.RegionOfInterestImageFilter()
    roi_filter.SetSize([end_index[i] - start_index[i] for i in range(3)])
    roi_filter.SetIndex(start_index)
    cropped_img = roi_filter.Execute(img)
    cropped_seg = roi_filter.Execute(seg)
    return cropped_img, cropped_seg

def marginal_crop_img_seg(img, seg, margin_mm):
    seg_array = sitk.GetArrayFromImage(seg)
    z_indices, y_indices, x_indices = seg_array.nonzero()
    if len(z_indices) == 0:
        raise ValueError("Segmentation is empty (no nonzero voxels).")

    z_min, z_max = int(z_indices.min()), int(z_indices.max())
    y_min, y_max = int(y_indices.min()), int(y_indices.max())
    x_min, x_max = int(x_indices.min()), int(x_indices.max())

    spacing = img.GetSpacing()  # (x,y,z)
    margin_voxels = [int(margin_mm / spacing[i]) for i in range(3)]

    x_min = max(0, x_min - margin_voxels[0])
    x_max = min(img.GetSize()[0] - 1, x_max + margin_voxels[0])
    y_min = max(0, y_min - margin_voxels[1])
    y_max = min(img.GetSize()[1] - 1, y_max + margin_voxels[1])
    z_min = max(0, z_min - margin_voxels[2])
    z_max = min(img.GetSize()[2] - 1, z_max + margin_voxels[2])

    start_index = [int(x_min), int(y_min), int(z_min)]  # ensure python int
    size = [int(x_max - x_min + 1), int(y_max - y_min + 1), int(z_max - z_min + 1)]

    roi_filter = sitk.RegionOfInterestImageFilter()
    roi_filter.SetSize(size)
    roi_filter.SetIndex(start_index)
    cropped_img = roi_filter.Execute(img)
    cropped_seg = roi_filter.Execute(seg)
    return cropped_img, cropped_seg

def resample_and_crop(img, seg, resample_spacing_mm, crop_dimensions_mm, marginal_crop:bool):
    resampled_img = resample_img_seg(img, new_spacing=resample_spacing_mm, is_label=False)
    resampled_seg = resample_img_seg(seg, new_spacing=resample_spacing_mm, is_label=True)
    # crop with fixed size around center of segmentation
    # cropped_img, cropped_seg = crop_img_seg(resampled_img, resampled_seg, crop_dimensions_mm)
    # marginal crop with 6 mm margin
    if marginal_crop:
        cropped_img, cropped_seg = marginal_crop_img_seg(resampled_img, resampled_seg, margin_mm=6.0)
    else:
        cropped_img, cropped_seg = crop_img_seg(resampled_img, resampled_seg, crop_dimensions_mm)
    return cropped_img, cropped_seg

def perform_one_case(args):
    img_path, seg_path, resample_spacing_mm, crop_dimensions_mm, output_images_dir, output_segmentations_dir, marginal_crop = args
    # check if output already exists
    if (output_images_dir / img_path.name).exists() and (output_segmentations_dir / seg_path.name).exists():
        print(f"Skipping {img_path.name} as output already exists.")
        return
    img = read_3d_volume(str(img_path))
    seg = read_3d_volume(str(seg_path))

    corrected_img = correct_orientation(img, target_orientation="RAS")
    corrected_seg = correct_orientation(seg, target_orientation="RAS")
    
    # check if segmentation is empty
    seg_array = sitk.GetArrayFromImage(corrected_seg)
    if np.sum(seg_array) == 0:
        print(f"Skipping {img_path.name} as segmentation is empty.")
        return  # skip empty segmentations
    final_img, final_seg = resample_and_crop(corrected_img, corrected_seg, resample_spacing_mm, crop_dimensions_mm, marginal_crop)
    sitk.WriteImage(final_img, str(output_images_dir / img_path.name))
    sitk.WriteImage(final_seg, str(output_segmentations_dir / seg_path.name))

def main(data_config_dir, marginal_crop):
    data_config = read_yaml(data_config_dir)
    preprocessed_data_base_dir = data_config["preprocessed_data_base_dir"]
    raw_images_dir = data_config['raw_images_dir']
    liver_nodes_instances_dir = Path(preprocessed_data_base_dir) / "02_liver_nodes_instances_segmentations_raw_data"
    resample_spacing_mm = data_config['resample_spacing_mm']
    crop_dimensions_mm = data_config['crop_dimensions_mm']

    img_groups_dir = Path(preprocessed_data_base_dir) / "img_groups.json"
    img_groups = read_json(img_groups_dir)
    all_segs = []
    all_imgs = []
    for group in img_groups:
        all_segs.extend(group["follow_ups_segs"] + [group["base_seg"]])
        all_imgs.extend(group["follow_ups_imgs"] + [group["base_img"]])

    output_dir = Path(data_config['preprocessed_data_base_dir'])
    if marginal_crop:
        print("Using marginal cropping strategy with 6 mm margin.")
        output_images_dir = output_dir / "04_images_resampled_marginal_cropped"
        output_segmentations_dir = output_dir / "04_segmentations_resampled_marginal_cropped"
    else:
        print("Using fixed-size cropping strategy with dimensions:", crop_dimensions_mm)
        output_images_dir = output_dir / "04_images_resampled_fixed_cropped"
        output_segmentations_dir = output_dir / "04_segmentations_resampled_fixed_cropped"
    output_segmentations_dir.mkdir(parents=True, exist_ok=True)
    output_images_dir.mkdir(parents=True, exist_ok=True)

    tasks = []
    for img_id, seg_id in tqdm(zip(all_imgs, all_segs)):

        img_path = Path(raw_images_dir) / img_id
        seg_path = Path(liver_nodes_instances_dir) / seg_id

        tasks.append((img_path, seg_path, resample_spacing_mm, crop_dimensions_mm, output_images_dir, output_segmentations_dir, marginal_crop))

    # perform_one_case(tasks[100])  # for debugging

    with ProcessPoolExecutor(max_workers=2) as executor:
        list(tqdm(executor.map(perform_one_case, tasks), total=len(tasks)))

if __name__ == "__main__":
    data_config_dir = '../configs/data_config.yaml'
    main(data_config_dir, marginal_crop=True)