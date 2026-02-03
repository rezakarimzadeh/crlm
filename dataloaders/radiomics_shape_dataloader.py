import pandas as pd
import numpy as np
from pathlib import Path
from utils import read_yaml, read_json
from radiomics_shape_dataset_prep import CustomDataset, FastCustomDataset
import torch
from torch.utils.data import DataLoader
import numpy as np


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
                matched_rows.append(match)
        return pd.concat(matched_rows).reset_index(drop=True)

    train_df = group_matched_indices(df, fold_img_groups['train'])
    val_df = group_matched_indices(df, fold_img_groups['val'])
    test_df = group_matched_indices(df, fold_img_groups['test'])
    return train_df, val_df, test_df

    

def collate_fn(batch):
    """
    batch: list of dicts returned by CustomDataset.__getitem__().
    Pads variable-length lesion sequences for base and follow-up separately.
    """

    def to_float_tensor(x):
        if isinstance(x, torch.Tensor):
            return x.float()
        if isinstance(x, np.ndarray):
            return torch.from_numpy(x).float()
        # list -> tensor
        return torch.tensor(x, dtype=torch.float32)

    def to_long_tensor(x):
        if isinstance(x, torch.Tensor):
            return x.long()
        if isinstance(x, np.ndarray):
            return torch.from_numpy(x).long()
        return torch.tensor(x, dtype=torch.long)

    def pad_lesion_set(features_list, labels_list=None, pad_value_feat=0.0, pad_value_lab=-1):
        """
        features_list: list of [Ti, F] tensors
        labels_list:   list of [Ti] tensors (optional)
        Returns:
          feat_out: [B, Tmax, F]
          pad_mask: [B, Tmax]  True where PAD
          lab_out:  [B, Tmax]  (or None)
        """
        B = len(features_list)
        Tmax = max(f.shape[0] for f in features_list)
        F = features_list[0].shape[1]

        feat_out = features_list[0].new_full((B, Tmax, F), fill_value=pad_value_feat)
        pad_mask = torch.ones((B, Tmax), dtype=torch.bool)  # start as all PAD

        lab_out = None
        if labels_list is not None:
            lab_out = torch.full((B, Tmax), fill_value=pad_value_lab, dtype=torch.long)

        for i, f in enumerate(features_list):
            Ti = f.shape[0]
            feat_out[i, :Ti] = f
            pad_mask[i, :Ti] = False  # False = real token, True = PAD

            if lab_out is not None:
                lab_out[i, :Ti] = labels_list[i]

        return feat_out, pad_mask, lab_out

    # -------------------------
    # Collect per-patient items
    # -------------------------
    pids = [item["patient_id"] for item in batch]

    base_feats = [to_float_tensor(item["base_img_features"]) for item in batch]
    base_lesion_labels = [to_long_tensor(item["base_lesion_labels"]) for item in batch]

    follow_feats = [to_float_tensor(item["followup_img_features"]) for item in batch]
    follow_lesion_labels = [to_long_tensor(item["followup_lesion_labels"]) for item in batch]

    demographic_info = torch.stack([to_float_tensor(item["demographic_info"]) for item in batch], dim=0)

    early_response = torch.tensor(
        [int(item["early_response"]) for item in batch],
        dtype=torch.long
    )
    overall_survival_24m = torch.tensor(
        [int(item["overall_survival_24m"]) for item in batch],
        dtype=torch.long
    )

    # -------------------------
    # Pad base & follow-up sets
    # -------------------------
    base_feat_pad, base_pad_mask, base_lab_pad = pad_lesion_set(base_feats, base_lesion_labels)
    follow_feat_pad, follow_pad_mask, follow_lab_pad = pad_lesion_set(follow_feats, follow_lesion_labels)

    # Optional passthrough of clinical dicts (kept as python objects)
    clinical_info = [item.get("all clinical_info", None) for item in batch]

    return {
        "pids": pids,

        "base": {
            "features": base_feat_pad,             # [B, Tbase, F]
            "pad_mask": base_pad_mask,             # [B, Tbase], True=PAD
            "lesion_labels": base_lab_pad,         # [B, Tbase], -1=PAD
        },

        "followup": {
            "features": follow_feat_pad,           # [B, Tfu, F]
            "pad_mask": follow_pad_mask,           # [B, Tfu], True=PAD
            "lesion_labels": follow_lab_pad,       # [B, Tfu], -1=PAD
        },

        "demographic_info": demographic_info,      # [B, D]
        "targets": {
            "early_response": early_response,      # [B]
            "overall_survival_24m": overall_survival_24m,  # [B]
        },

        "clinical_info": clinical_info,
    }

def get_radiomics_shape_dataloaders(data_config_dir, model_config_dir, feature_to_include, fold_idx):
    data_config = read_yaml(data_config_dir)
    excel_path = data_config["excel_path"]
    excel_table = read_excel_file_and_filter(excel_path)
    dataloader_config = read_yaml(model_config_dir)

    preprocessed_data_base_dir = data_config["preprocessed_data_base_dir"]
    fold_img_groups_path = Path(preprocessed_data_base_dir) / "five_fold_cv_splits" / f"five_fold_cv_split_{fold_idx}.json"
    fold_img_groups = read_json(fold_img_groups_path)
    matched_train_df, matched_val_df, matched_test_df = match_excel_splits_with_imgroups(excel_table, fold_img_groups)  
    print(f"Fold {fold_idx}: Train={len(matched_train_df)}, Val={len(matched_val_df)}, Test={len(matched_test_df)}")
    dataset_train = FastCustomDataset(matched_train_df, feature_to_include=feature_to_include, dataloader_config = dataloader_config)
    dataset_val = FastCustomDataset(matched_val_df, feature_to_include=feature_to_include, dataloader_config = dataloader_config)
    dataset_test = FastCustomDataset(matched_test_df, feature_to_include=feature_to_include, dataloader_config = dataloader_config)

    train_loader = DataLoader(dataset_train, batch_size=dataloader_config["batch_size"], shuffle=True, num_workers=6, collate_fn=collate_fn)
    val_loader = DataLoader(dataset_val, batch_size=dataloader_config["batch_size"], shuffle=False, num_workers=4, collate_fn=collate_fn)
    test_loader = DataLoader(dataset_test, batch_size=dataloader_config["batch_size"], shuffle=False, num_workers=4, collate_fn=collate_fn)
    return train_loader, val_loader, test_loader
    

def fn_test_loader(loader):
    print(f"Train dataset: {len(loader.dataset)} patients.")
    sample_data = next(iter(loader))
    print(f"Example batch keys: {list(sample_data.keys())}")
    print(f"Example base features shape: {sample_data['base']['features'].shape}")
    print(f"Example follow-up features shape: {sample_data['followup']['features'].shape}")
    print(f"Example demographic info shape: {sample_data['demographic_info'].shape}")
    print(f"Example early response targets: {sample_data['targets']['early_response']}")
    print(f"Example overall survival 24m targets: {sample_data['targets']['overall_survival_24m']}")

if __name__ == "__main__":
    data_config_dir = '../configs/data_config.yaml'
    model_config_dir = '../configs/radiomics_shape_model_config.yaml'
    fold_idx = 0  # Example fold index
    feature_to_include = ['shape', 'boundary', 'firstorder', 'glcm', 'glszm_glrlm']  # Example feature to include
    train_loader, val_loader, test_loader = get_radiomics_shape_dataloaders(data_config_dir, model_config_dir, feature_to_include, fold_idx)
    fn_test_loader(train_loader)