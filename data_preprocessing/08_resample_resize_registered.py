import numpy as np
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    Spacingd,
    ResizeWithPadOrCropd,
    Resized,
    EnsureTyped,
    SaveImaged,
)
from monai.data import Dataset
from utils import read_json, read_yaml


def build_preproc(resample_spacing_mm=(1.0, 1.0, 3.0), new_size=(192, 192, 128), device=None):
    # device=None keeps it simple (CPU). You can set device="cuda" later if desired.
    return Compose([
        LoadImaged(keys=("img", "seg"), reader="ITKReader", image_only=False),
        EnsureChannelFirstd(keys=("img", "seg")),
        Orientationd(keys=("img", "seg"), axcodes="RAS"),
        Spacingd(
            keys=("img", "seg"),
            pixdim=resample_spacing_mm,
            mode=("bilinear", "nearest"),
        ),
        Resized(keys=("img", "seg"), spatial_size=new_size, mode=("bilinear", "nearest")),
        EnsureTyped(keys=("img", "seg"), device=device, track_meta=True),
    ])


def perform_one_case(args):
    img_path, seg_path, tx, output_images_dir, output_segmentations_dir = args

    data = {"img": str(img_path), "seg": str(seg_path)}
    out = tx(data)

    # skip empty seg (after preprocessing, consistent with your original intent)
    seg_arr = out["seg"][0].detach().cpu().numpy()  # [1, Z, Y, X]
    if np.sum(seg_arr) == 0:
        print(f"Skipping {img_path.name} as segmentation is empty.")
        return

    # Save with MONAI to preserve affine/meta (direction/origin/spacing)
    SaveImaged(
        keys="img",
        output_dir=str(output_images_dir),
        output_postfix="",
        output_ext=".nii.gz",
        separate_folder=False,
        resample=False,
    )(out)

    SaveImaged(
        keys="seg",
        output_dir=str(output_segmentations_dir),
        output_postfix="",
        output_ext=".nii.gz",
        separate_folder=False,
        resample=False,
    )(out)


def main(data_config_dir):
    data_config = read_yaml(data_config_dir)
    base_dir = Path(data_config["preprocessed_data_base_dir"])

    images_dir = base_dir / "07_images_rigid_registration_cropped_to_base"
    segs_dir   = base_dir / "07_segmentations_rigid_registration_cropped_to_base"

    resample_spacing_mm = (1.0, 1.0, 3.0)
    new_size = (192, 192, 128)

    img_groups = read_json(base_dir / "img_groups.json")
    all_segs, all_imgs = [], []
    for group in img_groups:
        all_segs.extend(group["follow_ups_segs"] + [group["base_seg"]])
        all_imgs.extend(group["follow_ups_imgs"] + [group["base_img"]])

    output_images_dir = base_dir / f"08_images_resampled113_resized_{new_size[0]}_{new_size[1]}_{new_size[2]}"
    output_segmentations_dir = base_dir / f"08_segmentations_resampled113_resized_{new_size[0]}_{new_size[1]}_{new_size[2]}"
    output_images_dir.mkdir(parents=True, exist_ok=True)
    output_segmentations_dir.mkdir(parents=True, exist_ok=True)

    tx = build_preproc(resample_spacing_mm=resample_spacing_mm, new_size=new_size)

    tasks = []
    for img_id, seg_id in tqdm(list(zip(all_imgs, all_segs))):
        img_path = images_dir / img_id
        seg_path = segs_dir / seg_id
        # check if files exist
        if not img_path.exists() or not seg_path.exists():
            print(f"Warning: Missing files for {img_id} or {seg_id}. Skipping.")
            continue
        tasks.append((img_path, seg_path, tx, output_images_dir, output_segmentations_dir))

    # debug single case
    # perform_one_case(tasks[100])

    # full run
    with ProcessPoolExecutor(max_workers=10) as executor:
        list(tqdm(executor.map(perform_one_case, tasks), total=len(tasks)))


if __name__ == "__main__":
    main("../configs/data_config.yaml")
