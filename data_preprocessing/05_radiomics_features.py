import numpy as np
import SimpleITK as sitk
import pandas as pd
from radiomics import featureextractor
from utils import read_yaml
from pathlib import Path
import radiomics
radiomics.logger.setLevel("ERROR")
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm


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

def lesion_volume_mm3(mask, label: int) -> float:
    arr = sitk.GetArrayFromImage(mask)
    n = int(np.sum(arr == label))
    sx, sy, sz = mask.GetSpacing()  # (x,y,z)
    return float(n * sx * sy * sz)

def build_extractor(bin_width=25):
    params = {
        "setting": {
            "resampledPixelSpacing": None,
            "binWidth": float(bin_width),
            "normalize": False,
            "correctMask": True,
            "geometryTolerance": 1e-3,
        },
        "imageType": {"Original": {}},
        "featureClass": {
            "shape": [],
            "firstorder": [],
            "glcm": [],
            "glszm": [],
            "glrlm": [],
        },
    }
    return featureextractor.RadiomicsFeatureExtractor(params)




def extract_lesion_radiomics(
    ct_path: str,
    mask_path: str,
    bin_width: int = 25,
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

    extractor = build_extractor(bin_width=bin_width)

    lesion_rows = []
    for lab in labels:
        nvox = int(np.sum(arr == lab))
        if nvox < min_voxels:
            continue

        vol_mm3 = lesion_volume_mm3(msk, lab)

        # Execute radiomics for this label
        out = extractor.execute(img, msk, label=lab)
        feats = {k: v for k, v in out.items() if not k.startswith("diagnostics")}

        # Add metadata
        feats["lesion_label"] = lab
        feats["lesion_voxels"] = nvox
        feats["lesion_volume_mm3"] = vol_mm3
        lesion_rows.append(feats)

    if len(lesion_rows) == 0:
        raise ValueError(f"All lesions were below min_voxels={min_voxels}.")

    df_lesions = pd.DataFrame(lesion_rows)
    df_lesions.to_csv(out_csv_path, index=False)


def perform_one_extraction(args):
    ct_path, mask_path, bin_width, min_voxels, out_csv_path = args
    df_lesions = extract_lesion_radiomics(
                str(ct_path),
                str(mask_path),
                bin_width=bin_width,
                min_voxels=min_voxels,
                out_csv_path=out_csv_path,
            )


def main(data_config_dir):
    data_config = read_yaml(data_config_dir)

    preprocessed_data_base_dir = data_config["preprocessed_data_base_dir"]
    ct_base_dir = Path(preprocessed_data_base_dir) / "04_images_resampled_marginal_cropped"
    seg_base_dir = Path(preprocessed_data_base_dir) / "04_segmentations_resampled_marginal_cropped"

    output_dir = Path(preprocessed_data_base_dir) / "05_radiomics_features"
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
        out_csv_path = output_dir / f"{img_id.replace('.nii', '_radiomics.csv')}"
        tasks.append((str(ct_path), str(seg_path), data_config.get("radiomics_bin_width", 25), data_config.get("radiomics_min_voxels", 50), str(out_csv_path)))
        # try:
        #     df_lesions = extract_lesion_radiomics(
        #         str(ct_path),
        #         str(seg_path),
        #         bin_width=data_config.get("radiomics_bin_width", 25),
        #         min_voxels=data_config.get("radiomics_min_voxels", 50),
        #     )
        #     df_lesions.to_csv(out_csv_path, index=False)
        #     # print(f"Saved radiomics for {img_id} to {out_csv_path}.")
        # except Exception as e:
        #     print(f"Error processing {img_id}: {e}")
        # exit()
    
    with ProcessPoolExecutor(max_workers=10) as executor:
        list(tqdm(executor.map(perform_one_extraction, tasks), total=len(tasks)))
if __name__ == "__main__":
    data_config_dir = '../configs/data_config.yaml'
    main(data_config_dir)