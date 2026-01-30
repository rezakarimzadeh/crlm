from utils import read_3d_volume, read_json, read_yaml, save_to_json
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import os
from pathlib import Path
import SimpleITK as sitk
import json
import numpy as np
import matplotlib.pyplot as plt


def spacing_and_liver_size_check(seg_img_dir):
    seg_img = read_3d_volume(seg_img_dir)
    # change the orientation to RAS if needed
    seg_img = sitk.DICOMOrient(seg_img, "RAS")
    spacing = seg_img.GetSpacing()

    seg_array = sitk.GetArrayFromImage(seg_img)
    # merge all the labels bigger than 0 to have a unified segmentation
    seg_array = (seg_array > 0).astype(seg_array.dtype)

    # find the bounding box coordinates for the segmentation
    z_indices, y_indices, x_indices = seg_array.nonzero()

    if len(z_indices) == 0:
        return {
            "spacing": spacing,
            "liver_size_mm3": 0.0,
            "coordinates": {"z": (0, 0), "y": (0, 0), "x": (0, 0)},
            "liver_size_each_dimension_mm": {"z": 0.0, "y": 0.0, "x": 0.0}
        }
    
    z_min, z_max = z_indices.min(), z_indices.max()
    y_min, y_max = y_indices.min(), y_indices.max()
    x_min, x_max = x_indices.min(), x_indices.max()
    
    # calculate the liver size in mm3
    liver_size_each_dimension_mm = {
        'z': (z_max - z_min + 1) * spacing[2],
        'y': (y_max - y_min + 1) * spacing[1],
        'x': (x_max - x_min + 1) * spacing[0],
    }
    liver_size_mm3 = np.sum(seg_array) * spacing[0] * spacing[1] * spacing[2]
    output = {
    'spacing': tuple(float(s) for s in spacing),
    'liver_size_mm3': float(liver_size_mm3),
    'coordinates': {
        'z': (int(z_min), int(z_max)),
        'y': (int(y_min), int(y_max)),
        'x': (int(x_min), int(x_max)),
    },
    'liver_size_each_dimension_mm': liver_size_each_dimension_mm
    }
    
    return output


def _process_one_seg(args):
    seg_img_dir, output_dir, img_id = args
    image_infos = spacing_and_liver_size_check(seg_img_dir)
    # Save the spacing and liver size information to a JSON file
    output_json_path = Path(output_dir) / img_id.replace('.nii.gz', '_spacing_liversize.json')
    with open(output_json_path, 'w') as f:
        json.dump(image_infos, f, indent=4)


def main(data_config_dir):
    data_config = read_yaml(data_config_dir)
    preprocessed_data_base_dir = data_config["preprocessed_data_base_dir"]

    img_groups_path = Path(preprocessed_data_base_dir) / "img_groups.json"
    output_instances_dir = Path(preprocessed_data_base_dir) / "02_liver_nodes_instances_segmentations_raw_data"
    output_dir = Path(preprocessed_data_base_dir) / "03_check_spacing_liversize"
    output_dir.mkdir(parents=True, exist_ok=True)
    img_groups = read_json(img_groups_path)

    tasks = []
    for group in tqdm(img_groups):
        all_segs = group["follow_ups_segs"] + [group["base_seg"]]
        for img_id in all_segs:
            seg_path = Path(output_instances_dir) / img_id
            tasks.append((str(seg_path), str(output_dir), img_id))


    with ProcessPoolExecutor(max_workers=6) as executor:
        list(tqdm(executor.map(_process_one_seg, tasks), total=len(tasks)))

    return output_dir

def aggregate_results(output_dir):
    output_dir = Path(output_dir)
    json_files = list(output_dir.glob('*_spacing_liversize.json'))
    image_infos = {}
    for json_file in tqdm(json_files):
        info = read_json(str(json_file))
        image_infos[json_file.stem] = info
    
    save_to_json_path = output_dir / 'aggregated' / "aggregated_spacing_liversize.json"
    save_to_json_path.parent.mkdir(parents=True, exist_ok=True)
    save_to_json(image_infos, str(save_to_json_path))

    # find the max and min liver sizes across all images
    liver_sizes = [info['liver_size_mm3'] for info in image_infos.values()]
    liver_sizes_array = np.array(liver_sizes)
    min_liver_size = min(liver_sizes_array)
    max_liver_size = max(liver_sizes_array)
    print(f"Minimum Liver Size (mm3): {min_liver_size}")
    print(f"Maximum Liver Size (mm3): {max_liver_size}")

    # find the min and max and median spacing across all images
    spacings = [info['spacing'] for info in image_infos.values()]
    spacings_array = np.array(spacings)
    min_spacing = spacings_array.min(axis=0)
    max_spacing = spacings_array.max(axis=0)
    median_spacing = np.median(spacings_array, axis=0)
    print(f"Minimum Spacing (mm): {min_spacing}")
    print(f"Maximum Spacing (mm): {max_spacing}")
    print(f"Median Spacing (mm): {median_spacing}")
    
    # find min, max, median liver size in each dimension
    liver_sizes_each_dimension = [info['liver_size_each_dimension_mm'] for info in image_infos.values()]
    liver_sizes_each_dimension_array = {
        'z': np.array([size['z'] for size in liver_sizes_each_dimension]),
        'y': np.array([size['y'] for size in liver_sizes_each_dimension]),
        'x': np.array([size['x'] for size in liver_sizes_each_dimension]),
    }
    for dim in ['z', 'y', 'x']:
        min_size = liver_sizes_each_dimension_array[dim].min()
        max_size = liver_sizes_each_dimension_array[dim].max()
        median_size = np.median(liver_sizes_each_dimension_array[dim])
        print(f"Dimension {dim}: Min Size (mm): {min_size}, Max Size (mm): {max_size}, Median Size (mm): {median_size}")

    aggregated_stats = {
        "min_liver_size_mm3": float(min_liver_size),
        "max_liver_size_mm3": float(max_liver_size),
        "min_spacing_mm": tuple(float(s) for s in min_spacing),
        "max_spacing_mm": tuple(float(s) for s in max_spacing),
        "median_spacing_mm": tuple(float(s) for s in median_spacing),
        "min_liver_size_each_dimension_mm": {dim: float(liver_sizes_each_dimension_array[dim].min()) for dim in ['z', 'y', 'x']},
        "max_liver_size_each_dimension_mm": {dim: float(liver_sizes_each_dimension_array[dim].max()) for dim in ['z', 'y', 'x']},
        "median_liver_size_each_dimension_mm": {dim: float(np.median(liver_sizes_each_dimension_array[dim])) for dim in ['z', 'y', 'x']}
    }
    save_to_json(aggregated_stats, str(output_dir / 'aggregated' / "aggregated_stats.json"))

    # Plot histogram of liver sizes each dimension mm
    plt.figure(figsize=(6, 18))
    for dim in ['z', 'y', 'x']:
        plt.subplot(3, 1, ['z', 'y', 'x'].index(dim) + 1)
        plt.hist(liver_sizes_each_dimension_array[dim], bins=100, alpha=0.5, label=f'Dimension {dim}')
        plt.xlabel('Liver Size (mm)')
        plt.ylabel('Frequency')
        plt.title(f'Histogram of Liver Sizes in Dimension {dim}')
        plt.legend()
    plt_path = output_dir / 'aggregated' / 'liver_size_histogram.png'
    plt.savefig(str(plt_path))
    plt.close()
    # plot histogram of spacing
    plt.figure(figsize=(6, 18))
    for dim in range(3):
        plt.subplot(3, 1, dim + 1)
        plt.hist(spacings_array[:, dim], bins=100, alpha=0.5, label=f'Spacing Dimension {dim}')
        plt.xlabel('Spacing (mm)')
        plt.ylabel('Frequency')
        plt.title(f'Histogram of Spacing in Dimension {dim}')
        plt.legend()
    plt_path = output_dir / 'aggregated' / 'spacing_histogram.png'
    plt.savefig(str(plt_path))
    plt.close()

if __name__ == "__main__":
    data_config_dir = '../configs/data_config.yaml'
    # output_dir = main(data_config_dir)
    output_dir = Path('/mnt/l/Basic/divi/jstoker/slicer_pdac/Marius/Reza Morphology/data/preprocessed_data/CRLM/03_check_spacing_liversize')
    aggregate_results(output_dir)