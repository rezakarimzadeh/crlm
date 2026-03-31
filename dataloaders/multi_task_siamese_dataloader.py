import os
import pandas as pd
import numpy as np
from pathlib import Path
from utils import read_yaml, read_json
import torch
import numpy as np
from monai.transforms import (
    AsDiscrete,
    EnsureChannelFirstd,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandFlipd,
    RandCropByPosNegLabeld,
    RandShiftIntensityd,
    ScaleIntensityRanged,
    Spacingd,
    RandRotate90d,
    Resized,
    RandAffined,
    EnsureTyped,
    RandShiftIntensityd,
    MapTransform,
)
from monai.data import DataLoader, Dataset
from monai.data.image_reader import NibabelReader
import nibabel as nib


class MergeLabelsGE2ToOneD(MapTransform):
    """
    Dictionary-based transform:
    Convert segmentation labels >= 2 to 1, else 0.
    """
    def __init__(self, keys):
        super().__init__(keys)

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            seg = d[key]
            
            if not isinstance(seg, torch.Tensor):
                seg = torch.as_tensor(seg)

            d[key] = (seg >= 2).to(seg.dtype)

        return d

def read_excel_file_and_filter(excel_path):
    # Read the Excel file
    df = pd.read_excel(excel_path)
    filtered_df = df[df['Segmented'] == 'Yes'].reset_index(drop=True)
    return filtered_df


def match_excel_splits_with_imgroups(df, fold_img_groups):

    def group_matched_indices(df, group):
        matched_rows = []
        for case in group:
            patient_id = case['patient_id']
            patient_number = int("".join(filter(str.isdigit, patient_id)))
            match = df[df['SubjectKey'] == patient_number]
            if not match.empty:
                # add patient_id as a column
                match = match.copy()
                match['patient_id'] = patient_id
                match['early_recurrence'] = match['ER (1 = yes, 0 = no)'].astype(int)
                match['overall_survival_24m'] = (match['OSm'] > 24).astype(int)
                match['demographic_info'] = match[['mutstat_enc', 'sex_enc', 'who_enc', 'age_f', 'baseline_ttv', 'delta_ttv_rel']].values.tolist()
                match['pathology'] = match['pathology_enc'].astype(int)
                matched_rows.append(match)
        return pd.concat(matched_rows).reset_index(drop=True)
    
    mut_map = {
        "BRAF mutation": 0,
        "RAS & BRAF wildtype": 1,
        "RAS mutation": 2,
    }
    sex_map = {"Female": 0, "Male": 1}

    pathology_map = {"nan": -1,
                        "No histological response": 0,
                        "Partial histological response": 1,
                        "Major histological response": 1} 
    
    morph_response_map = {"No response": 0, "Optimal response": 1, "Suboptimal response": 2, "Unknown": -1}
    morphscore_map = {"Unknown": -1, 1:0, 2:1, 3:2}
    bevacizumab_map = {"No": 0, "Yes": 1}  # Assuming these are the only values, otherwise use .get() with default
    
    # Map / coerce
    df["mutstat_enc"] = df["mutstat"].map(mut_map).fillna(-1).astype(int)
    df["sex_enc"] = df["sex"].map(sex_map).fillna(-1).astype(int)
    df["pathology_enc"] = df["Pathology"].fillna("nan").map(pathology_map).astype(int)
    df["who_enc"] = pd.to_numeric(df["WHO"], errors="coerce").fillna(-1).astype(int)
    df["age_f"] = pd.to_numeric(df["Age"], errors="coerce").fillna(-1.0).astype(float)
    df["baseline_ttv"] = pd.to_numeric(df["Baseline volume ml"], errors="coerce").fillna(-1.0).astype(float)
    df["delta_ttv_rel"] = pd.to_numeric(df["FU1 delta vol rel"], errors="coerce").fillna(-1.0).astype(float)
    df["morph_response_enc"] = df["morphresponse_best"].map(morph_response_map).fillna(-1).astype(int)
    df["morph_score_base"] = df["morphscorebase_majority"].map(morphscore_map).fillna(-1).astype(int)
    df["morph_score_followup"] = df["morphscorefirstfu_majority"].map(morphscore_map).fillna(-1).astype(int)
    df["early_recurrence"] = pd.to_numeric(df["ER (1 = yes, 0 = no)"], errors="coerce").fillna(0).astype(int)
    df["bevacizumab"] = df["Bevacizumab"].map(bevacizumab_map).fillna(0).astype(int)
    
    train_df = group_matched_indices(df, fold_img_groups['train'])
    val_df = group_matched_indices(df, fold_img_groups['val'])
    test_df = group_matched_indices(df, fold_img_groups['test'])
    return train_df, val_df, test_df

class VolumesDataset(Dataset):
    def __init__(self, df, preprocessed_data_base_dir, train, dataloader_config):
        self.df = df
        self.preprocessed_data_base_dir = preprocessed_data_base_dir
        self.hu_window = tuple(dataloader_config["hu_window"])
        self.dataloader_config = dataloader_config
        self.transformations = self._transformations(train=train)

    def __len__(self):
        return len(self.df)
    
    def _transformations(self, train, img_keys=["base_img", "followup_img"], seg_keys=["base_seg", "followup_seg"], img_mode=["trilinear", "trilinear"], seg_mode=["nearest", "nearest"]):
        # 1) deterministic, shared
        pre = [
            LoadImaged(keys=img_keys + seg_keys, reader=NibabelReader()),
            EnsureChannelFirstd(keys=img_keys + seg_keys),
            Orientationd(keys=img_keys + seg_keys, axcodes="RAS"),
            MergeLabelsGE2ToOneD(keys=seg_keys),
            ScaleIntensityRanged(
                keys=img_keys,
                a_min=self.hu_window[0], a_max=self.hu_window[1],
                b_min=0.0, b_max=1.0, clip=True
            ),
        ]

        if not train:
            # return Compose(pre)
            return Compose(pre + [
                EnsureTyped(keys=img_keys + seg_keys, track_meta=False)
            ])

        # 2) coupled spatial augs (same random params for both keys)
        spatial = [
            RandFlipd(keys=img_keys + seg_keys, spatial_axis=0, prob=0.20),
            RandFlipd(keys=img_keys + seg_keys, spatial_axis=1, prob=0.20),
            RandFlipd(keys=img_keys + seg_keys, spatial_axis=2, prob=0.20),
            RandRotate90d(keys=img_keys + seg_keys, prob=0.20, max_k=3),
            RandAffined(
                keys=img_keys + seg_keys,
                mode=img_mode + seg_mode,  # per-key interpolation allowed
                prob=0.2,
                # spatial_size=(192, 192, 128),
                spatial_size=(96, 96, 64),
                rotate_range=(np.pi/18, np.pi/18, np.pi/18),
                scale_range=(0.1, 0.1, 0.1),
            ),
        ]

        # 3) intensity aug 
        # intensity = [
        #     RandShiftIntensityd(keys=img_keys, offsets=0.10, prob=0.10),
        # ]

        # return Compose(pre + spatial)
        return Compose(pre + spatial + [
                                EnsureTyped(keys=img_keys + seg_keys, track_meta=False)
                            ])

    def _get_available_idxs(self, idx):
        sample = self.df.iloc[idx]
        case = {
                "patient_ids": sample['patient_id'],
                # "base_img": os.path.join(self.preprocessed_data_base_dir,"10_images_no_registration_resampled113_resized_192_192_128", f"{sample['patient_id']}_0_0000.nii.gz"),
                # "base_seg": os.path.join(self.preprocessed_data_base_dir,"10_segmentations_no_registration_resampled113_resized_192_192_128", f"{sample['patient_id']}_0.nii.gz"),

                # "followup_img": os.path.join(self.preprocessed_data_base_dir,"10_images_no_registration_resampled113_resized_192_192_128", f"{sample['patient_id']}_1_0000.nii.gz"),
                # "followup_seg": os.path.join(self.preprocessed_data_base_dir,"10_segmentations_no_registration_resampled113_resized_192_192_128", f"{sample['patient_id']}_1.nii.gz"),

                "base_img": os.path.join(self.preprocessed_data_base_dir,"11_images_no_registration_resampled226_resized_96_96_64", f"{sample['patient_id']}_0_0000.nii.gz"),
                "base_seg": os.path.join(self.preprocessed_data_base_dir,"11_segmentations_no_registration_resampled226_resized_96_96_64", f"{sample['patient_id']}_0.nii.gz"),

                "followup_img": os.path.join(self.preprocessed_data_base_dir,"11_images_no_registration_resampled226_resized_96_96_64", f"{sample['patient_id']}_1_0000.nii.gz"),
                "followup_seg": os.path.join(self.preprocessed_data_base_dir,"11_segmentations_no_registration_resampled226_resized_96_96_64", f"{sample['patient_id']}_1.nii.gz"),

                "bevacizumab": torch.tensor(sample['bevacizumab']),
                "demographic_info": torch.tensor(sample['demographic_info']),  # Example demographic info, adjust as needed
                "targets": {"early_recurrence": torch.tensor(sample['early_recurrence']),
                            "overall_survival_24m": torch.tensor(sample['overall_survival_24m']),
                            "pathology": torch.tensor(sample['pathology']),
                            "morph_response": torch.tensor(sample['morph_response_enc']),
                            "morph_score_base": torch.tensor(sample['morph_score_base']),
                            "morph_score_followup": torch.tensor(sample['morph_score_followup']),
                            }
        }
        #  check if files exist
        if not os.path.exists(case["base_img"]) or not os.path.exists(case["followup_img"]) or not os.path.exists(case["base_seg"]) or not os.path.exists(case["followup_seg"]):
            new_idx = np.random.randint(0, len(self.df))
            return self._get_available_idxs(new_idx)
        else:            
            return case

    def __getitem__(self, idx):
        case = self._get_available_idxs(idx)
        case = self.transformations(case)
        return case
        

def print_label_statistics(prepared_dataset_df):
    print("Label distribution:")
    print(prepared_dataset_df["early_recurrence"].value_counts())
    print(prepared_dataset_df["overall_survival_24m"].value_counts())


def get_mtl_siamese_dataloaders(data_config_dir, model_config_dir, fold_idx):
    data_config = read_yaml(data_config_dir)
    excel_path = data_config["excel_path"]
    excel_table = read_excel_file_and_filter(excel_path)
    dataloader_config = read_yaml(model_config_dir)

    preprocessed_data_base_dir = data_config["preprocessed_data_base_dir"]
    fold_img_groups_path = Path(preprocessed_data_base_dir) / "five_fold_cv_splits" / f"five_fold_cv_split_{fold_idx}.json"
    fold_img_groups = read_json(fold_img_groups_path)
    matched_train_df, matched_val_df, matched_test_df = match_excel_splits_with_imgroups(excel_table, fold_img_groups)

    train_df = pd.concat([matched_train_df.copy(), matched_val_df.copy()], axis=0, ignore_index=True)
    val_df = matched_test_df.copy()
    test_df = matched_test_df.copy()

    print(f"Fold {fold_idx}: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
    print("Train label distribution:")
    print_label_statistics(train_df)
    print("Val label distribution:")
    print_label_statistics(val_df)
    print("Test label distribution:")
    print_label_statistics(test_df)
    dataset_train = VolumesDataset(train_df, preprocessed_data_base_dir=preprocessed_data_base_dir, train=True, dataloader_config=dataloader_config)
    dataset_val = VolumesDataset(val_df, preprocessed_data_base_dir=preprocessed_data_base_dir, train=False, dataloader_config=dataloader_config)
    dataset_test = VolumesDataset(test_df, preprocessed_data_base_dir=preprocessed_data_base_dir, train=False, dataloader_config=dataloader_config)

    train_loader = DataLoader(  dataset_train, batch_size=dataloader_config["batch_size"], shuffle=True, 
                                num_workers=6,
                                pin_memory=True,
                                persistent_workers=True,
                                prefetch_factor=2,
                              )
    val_loader = DataLoader(dataset_val, batch_size=dataloader_config["batch_size"], shuffle=False, 
                                num_workers=4,
                                pin_memory=True,
                                persistent_workers=True,
                                prefetch_factor=2,
                                )
    test_loader = DataLoader(dataset_test, batch_size=dataloader_config["batch_size"], shuffle=False, 
                                num_workers=2,
                                pin_memory=True,
                                persistent_workers=True,
                                prefetch_factor=2,
                                )
    return train_loader, val_loader, test_loader
    

def fn_test_loader(loader):
    print(f"Train dataset: {len(loader.dataset)} patients.")
    sample_data = next(iter(loader))
    print(f"Example batch keys: {list(sample_data.keys())}")
    print(f"Example base img shape: {sample_data['base_img'].shape}")
    print(f"Example follow-up img shape: {sample_data['followup_img'].shape}")

    print(f"example base seg shape: {sample_data['base_seg'].shape}")
    print(f"example follow-up seg shape: {sample_data['followup_seg'].shape}")
    print(f"max, min values in example base img: {sample_data['base_img'].max()}, {sample_data['base_img'].min()}")
    print(f"max, min values in example follow-up img: {sample_data['followup_img'].max()}, {sample_data['followup_img'].min()}")

    print(f"unique values in example base seg: {torch.unique(sample_data['base_seg'])}")
    print(f"max, min values in example base seg: {sample_data['base_seg'].max()}, {sample_data['base_seg'].min()}")
    print(f"unique values in example follow-up seg: {torch.unique(sample_data['followup_seg'])}")
    print(f"max, min values in example follow-up seg: {sample_data['followup_seg'].max()}, {sample_data['followup_seg'].min()}")
    
    print(f"Example early recurrence targets: {sample_data['targets']['early_recurrence']}")
    print(f"Example overall survival 24m targets: {sample_data['targets']['overall_survival_24m']}")

    print(f"Example pathology targets: {sample_data['targets']['pathology']}")
    print(f"Unique pathology targets in batch: {torch.unique(sample_data['targets']['pathology'])}")

    print(f"Example demographic info: {sample_data['demographic_info']}")
    print(f"Example demographic info shape: {sample_data['demographic_info'].shape}")
    
    # for batch in loader:
    #     print(f"Batch base features shape: {batch['base_img'].shape}")
    #     print(f"Batch follow-up features shape: {batch['followup_img'].shape}")

    # save example batch to disk for inspection as nii.gz files
    output_dir = Path("example_batch_output")
    output_dir.mkdir(exist_ok=True)
    for i in range(sample_data['base_img'].shape[0]):
        base_img = sample_data['base_img'][i].numpy().squeeze()
        followup_img = sample_data['followup_img'][i].numpy().squeeze()
        base_seg = sample_data['base_seg'][i].numpy().squeeze()
        followup_seg = sample_data['followup_seg'][i].numpy().squeeze()

        # save as nii.gz files
        nib.save(nib.Nifti1Image(base_img, affine=np.eye(4)), output_dir / f"example_base_img_{i}.nii.gz")
        nib.save(nib.Nifti1Image(followup_img, affine=np.eye(4)), output_dir / f"example_followup_img_{i}.nii.gz")
        nib.save(nib.Nifti1Image(base_seg, affine=np.eye(4)), output_dir / f"example_base_seg_{i}.nii.gz")
        nib.save(nib.Nifti1Image(followup_seg, affine=np.eye(4)), output_dir / f"example_followup_seg_{i}.nii.gz")

if __name__ == "__main__":
    data_config_dir = '../configs/data_config.yaml'
    model_config_dir = '../configs/cnn_config.yaml'
    fold_idx = 0  # Example fold index
    train_loader, val_loader, test_loader = get_mtl_siamese_dataloaders(data_config_dir, model_config_dir, fold_idx)
    fn_test_loader(train_loader)