from pathlib import Path
import SimpleITK as sitk
from utils import read_3d_volume, read_json, read_yaml
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import os


def select_liver_nodes_segmentations(seg_img):
    # Assuming liver 12 + nodes 13 => liver:1, nodes:2
    seg_array = sitk.GetArrayFromImage(seg_img)
    liver_array = (seg_array == 12).astype(seg_array.dtype)
    nodes_array = (seg_array == 13).astype(seg_array.dtype) * 2
    selected_array = liver_array + nodes_array
    selected_img = sitk.GetImageFromArray(selected_array)
    selected_img.CopyInformation(seg_img)
    return selected_img

# seperate liver nodes
def label_instances_cc(seg_img, connectivity26=True, node_label=13, liver_label=12):
    """
    Returns instance label image:
      background = 0
      liver      = 1
      nodes      = 2..K+1  (connected components of node_label, sorted by size)
    """
    # nodes (binary)
    nodes_bin = sitk.BinaryThreshold(seg_img, node_label, node_label, 1, 0)

    # connected components on nodes
    cc_filt = sitk.ConnectedComponentImageFilter()
    cc_filt.SetFullyConnected(bool(connectivity26))
    cc = cc_filt.Execute(nodes_bin)

    relabel = sitk.RelabelComponentImageFilter()
    relabel.SetSortByObjectSize(True)
    nodes_inst = relabel.Execute(cc)  # UInt32 labels: 0..K

    # shift node ids: 1..K -> 2..K+1
    nodes_inst = sitk.Cast(nodes_inst, sitk.sitkUInt32)
    nodes_shift = nodes_inst + sitk.Cast(nodes_inst > 0, sitk.sitkUInt32)

    # liver mask (binary)
    liver_mask = sitk.BinaryThreshold(seg_img, liver_label, liver_label, 1, 0)
    liver_mask_u32 = sitk.Cast(liver_mask, sitk.sitkUInt32)

    # start from nodes, then write liver=1 where liver mask is 1
    out = sitk.Mask(nodes_shift, liver_mask_u32 == 0)          # zero-out nodes inside liver region if you want
    out = out + liver_mask_u32 * 1                             # set liver to 1

    return out

# ---- worker (must be top-level for multiprocessing) ----
def _process_one_seg(args):
    raw_segmentations_ai_dir, output_dir, img_id, output_instances_dir = args

    seg_path = Path(raw_segmentations_ai_dir) / img_id
    out_path = Path(output_dir) / img_id
    out_inst_path = Path(output_instances_dir) / img_id
    
    seg_img = read_3d_volume(str(seg_path))
    liver_nodes_img = select_liver_nodes_segmentations(seg_img)
    try:
        inst_img = label_instances_cc(seg_img)
    except Exception as e:
        print(f"Error processing {img_id}: {e}")
        return None

    # Ensure parent exists (in case img_id has subfolders)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    sitk.WriteImage(liver_nodes_img, str(out_path))
    sitk.WriteImage(inst_img, str(out_inst_path))
    return img_id


def main(data_config_dir):
    data_config = read_yaml(data_config_dir)
    raw_segmentations_ai_dir = data_config["raw_segmentations_ai_dir"]
    preprocessed_data_base_dir = data_config["preprocessed_data_base_dir"]

    img_groups_path = Path(preprocessed_data_base_dir) / "img_groups.json"
    output_dir = Path(preprocessed_data_base_dir) / "02_liver_nodes_segmentations_raw_data"
    output_instances_dir = Path(preprocessed_data_base_dir) / "02_liver_nodes_instances_segmentations_raw_data"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_instances_dir.mkdir(parents=True, exist_ok=True)

    img_groups = read_json(img_groups_path)

    # ---- build a flat task list ----
    tasks = []
    for group in img_groups:
        all_segs = group["follow_ups_segs"] + [group["base_seg"]]
        for img_id in all_segs:
            tasks.append((raw_segmentations_ai_dir, str(output_dir), img_id, str(output_instances_dir)))

    # Optional: avoid duplicates if same img_id appears multiple times
    # (keeps first occurrence order)
    seen = set()
    uniq_tasks = []
    for t in tasks:
        if t[2] not in seen:
            uniq_tasks.append(t)
            seen.add(t[2])

    n_workers = 2# min(os.cpu_count() or 1, 16)  # cap if you want

    # ---- parallel execution ----
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        list(tqdm(ex.map(_process_one_seg, uniq_tasks), total=len(uniq_tasks)))


if __name__ == "__main__":
    data_config_path = "../configs/data_config.yaml"
    main(data_config_path)