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
)
from monai.data import DataLoader, Dataset
from monai.data.image_reader import NibabelReader


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
                match['early_response'] = match['ER (1 = yes, 0 = no)'].astype(int)
                match['overall_survival_24m'] = (match['OSm'] > 24).astype(int)
                matched_rows.append(match)
        return pd.concat(matched_rows).reset_index(drop=True)

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
        self.transformations = self._get_transformations(train=train)

    def __len__(self):
        return len(self.df)
    
    def _get_transformations(self, train, keys=["base_img", "followup_img"], interpolation_mode=("bilinear", "bilinear")):
        train_transforms = Compose(
                        [
                            LoadImaged(keys=keys, reader=NibabelReader()),
                            EnsureChannelFirstd(keys=keys),
                            Orientationd(keys=keys, axcodes="RAS"),
                            # Spacingd(
                            #     keys=keys,
                            #     pixdim=(1.0, 1.0, 3.0),
                            #     mode=interpolation_mode,
                            # ),
                            # Resized(keys=keys, spatial_size=self.dataloader_config["resize"], mode=interpolation_mode),
                            ScaleIntensityRanged(
                                keys=keys,
                                a_min=self.hu_window[0],
                                a_max=self.hu_window[1],
                                b_min=0.0,
                                b_max=1.0,
                                clip=True,
                            ),
                            RandFlipd(
                                keys=keys,
                                spatial_axis=[0],
                                prob=0.10,
                            ),
                            RandFlipd(
                                keys=keys,
                                spatial_axis=[1],
                                prob=0.10,
                            ),
                            RandFlipd(
                                keys=keys,
                                spatial_axis=[2],
                                prob=0.10,
                            ),
                            RandRotate90d(
                                keys=keys,
                                prob=0.10,
                                max_k=3,
                            ),
                        ]
                    )

        val_transforms = Compose(
                        [
                            LoadImaged(keys=keys, reader=NibabelReader()),
                            EnsureChannelFirstd(keys=keys),
                            Orientationd(keys=keys, axcodes="RAS"),
                            # Spacingd(
                            #     keys=keys,
                            #     pixdim=(1.0, 1.0, 3.0),
                            #     mode=interpolation_mode,
                            # ),
                            # Resized(keys=keys, spatial_size=self.dataloader_config["resize"], mode=interpolation_mode),
                            ScaleIntensityRanged(keys=keys, a_min=self.hu_window[0], a_max=self.hu_window[1], b_min=0.0, b_max=1.0, clip=True),
                        ]
                    )
        if train:
            return train_transforms
        else:
            return val_transforms

    def _get_available_idxs(self, idx):
        sample = self.df.iloc[idx]
        case = {
            "base_img": os.path.join(self.preprocessed_data_base_dir,"08_images_resampled113_resized_192_192_128", f"{sample['patient_id']}_0_0000.nii.gz"),
            "followup_img": os.path.join(self.preprocessed_data_base_dir,"08_images_resampled113_resized_192_192_128", f"{sample['patient_id']}_1_0000.nii.gz"),
            "early_response": torch.tensor(sample['early_response']),
            "overall_survival_24m": torch.tensor(sample['overall_survival_24m'])
        }
        #  check if files exist
        if not os.path.exists(case["base_img"]) or not os.path.exists(case["followup_img"]):
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
    print(prepared_dataset_df["early_response"].value_counts())
    print(prepared_dataset_df["overall_survival_24m"].value_counts())


def get_cnn_dataloaders(data_config_dir, model_config_dir, fold_idx):
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
    
    train_loader = DataLoader(dataset_train, batch_size=dataloader_config["batch_size"], shuffle=True, num_workers=8)
    val_loader = DataLoader(dataset_val, batch_size=dataloader_config["batch_size"], shuffle=False, num_workers=2)
    test_loader = DataLoader(dataset_test, batch_size=dataloader_config["batch_size"], shuffle=False, num_workers=2)
    return train_loader, val_loader, test_loader
    

def fn_test_loader(loader):
    print(f"Train dataset: {len(loader.dataset)} patients.")
    sample_data = next(iter(loader))
    print(f"Example batch keys: {list(sample_data.keys())}")
    print(f"Example base features shape: {sample_data['base_img'].shape}")
    print(f"Example follow-up features shape: {sample_data['followup_img'].shape}")
    print(f"Example early response targets: {sample_data['early_response']}")
    print(f"Example overall survival 24m targets: {sample_data['overall_survival_24m']}")
    for batch in loader:
        print(f"Batch base features shape: {batch['base_img'].shape}")
        print(f"Batch follow-up features shape: {batch['followup_img'].shape}")

if __name__ == "__main__":
    data_config_dir = '../configs/data_config.yaml'
    model_config_dir = '../configs/cnn_config.yaml'
    fold_idx = 0  # Example fold index
    train_loader, val_loader, test_loader = get_cnn_dataloaders(data_config_dir, model_config_dir, fold_idx)
    fn_test_loader(train_loader)