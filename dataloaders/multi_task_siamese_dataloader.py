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
                match['demographic_info'] = match[['mutstat_enc', 'sex_enc', 'who_enc', 'age_f']].values.tolist()
                matched_rows.append(match)
        return pd.concat(matched_rows).reset_index(drop=True)
    
    mut_map = {
        "BRAF mutation": 0,
        "RAS & BRAF wildtype": 1,
        "RAS mutation": 2,
    }
    sex_map = {"Female": 0, "Male": 1}

    # Map / coerce
    df["mutstat_enc"] = df["mutstat"].map(mut_map).fillna(-1).astype(int)
    df["sex_enc"] = df["sex"].map(sex_map).fillna(-1).astype(int)
    df["who_enc"] = pd.to_numeric(df["WHO"], errors="coerce").fillna(-1).astype(int)
    df["age_f"] = pd.to_numeric(df["Age"], errors="coerce").fillna(-1.0).astype(float)
    
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
            return Compose(pre)

        # 2) coupled spatial augs (same random params for both keys)
        spatial = [
            RandFlipd(keys=img_keys + seg_keys, spatial_axis=0, prob=0.10),
            RandFlipd(keys=img_keys + seg_keys, spatial_axis=1, prob=0.10),
            RandFlipd(keys=img_keys + seg_keys, spatial_axis=2, prob=0.10),
            RandRotate90d(keys=img_keys + seg_keys, prob=0.10, max_k=3),
            RandAffined(
                keys=img_keys + seg_keys,
                mode=img_mode + seg_mode,  # per-key interpolation allowed
                prob=0.1,
                spatial_size=(192, 192, 128),
                rotate_range=(np.pi/18, np.pi/18, np.pi/18),
                scale_range=(0.1, 0.1, 0.1),
            ),
        ]

        # 3) intensity aug 
        # intensity = [
        #     RandShiftIntensityd(keys=img_keys, offsets=0.10, prob=0.10),
        # ]

        return Compose(pre + spatial)

    def _get_available_idxs(self, idx):
        sample = self.df.iloc[idx]
        case = {
                "base_img": os.path.join(self.preprocessed_data_base_dir,"10_images_no_registration_resampled113_resized_192_192_128", f"{sample['patient_id']}_0_0000.nii.gz"),
                "base_seg": os.path.join(self.preprocessed_data_base_dir,"10_segmentations_no_registration_resampled113_resized_192_192_128", f"{sample['patient_id']}_0.nii.gz"),

                "followup_img": os.path.join(self.preprocessed_data_base_dir,"10_images_no_registration_resampled113_resized_192_192_128", f"{sample['patient_id']}_1_0000.nii.gz"),
                "followup_seg": os.path.join(self.preprocessed_data_base_dir,"10_segmentations_no_registration_resampled113_resized_192_192_128", f"{sample['patient_id']}_1.nii.gz"),
                
                "demographic_info": torch.tensor(sample['demographic_info']),  # Example demographic info, adjust as needed
                "targets": {"early_recurrence": torch.tensor(sample['early_recurrence']),
                            "overall_survival_24m": torch.tensor(sample['overall_survival_24m'])
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
    print(f"Fold {fold_idx}: Train={len(matched_train_df)}, Val={len(matched_val_df)}, Test={len(matched_test_df)}")
    print("Train label distribution:")
    print_label_statistics(matched_train_df)
    print("Val label distribution:")
    print_label_statistics(matched_val_df)
    print("Test label distribution:")
    print_label_statistics(matched_test_df)
    dataset_train = VolumesDataset(matched_train_df, preprocessed_data_base_dir=preprocessed_data_base_dir, train=True, dataloader_config=dataloader_config)
    dataset_val = VolumesDataset(matched_val_df, preprocessed_data_base_dir=preprocessed_data_base_dir, train=False, dataloader_config=dataloader_config)
    dataset_test = VolumesDataset(matched_test_df, preprocessed_data_base_dir=preprocessed_data_base_dir, train=False, dataloader_config=dataloader_config)
    
    train_loader = DataLoader(dataset_train, batch_size=dataloader_config["batch_size"], shuffle=True, num_workers=4)
    val_loader = DataLoader(dataset_val, batch_size=dataloader_config["batch_size"], shuffle=False, num_workers=4)
    test_loader = DataLoader(dataset_test, batch_size=dataloader_config["batch_size"], shuffle=False, num_workers=4)
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
    # for batch in loader:
    #     print(f"Batch base features shape: {batch['base_img'].shape}")
    #     print(f"Batch follow-up features shape: {batch['followup_img'].shape}")

if __name__ == "__main__":
    data_config_dir = '../configs/data_config.yaml'
    model_config_dir = '../configs/cnn_config.yaml'
    fold_idx = 0  # Example fold index
    train_loader, val_loader, test_loader = get_mtl_siamese_dataloaders(data_config_dir, model_config_dir, fold_idx)
    fn_test_loader(train_loader)