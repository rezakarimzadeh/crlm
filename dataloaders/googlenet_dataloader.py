from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from torchvision import transforms
from utils import read_yaml, read_json
from glob import glob
import os
import pandas as pd
from tqdm import tqdm
from torchvision.utils import make_grid


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

   
class GooglenetDataset(Dataset):
    def __init__(self, matched_df, data_dir, train):
        self.data_dir = Path(data_dir) / "09_googlenet_2d_slices"
        self.matched_df = matched_df
        self.load_data_in_memory()  # Preload all data into memory for faster access during training/validation/testing 
        if train:
            # rotation, shift, scaling 
            self.transform = transforms.Compose([
                transforms.RandomRotation(degrees=30),
                transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
                transforms.RandomResizedCrop(size=224, scale=(0.95, 1.05)),
                transforms.ToTensor(),  # converts to [0,1]
                transforms.Normalize(mean=[0.485,0.456,0.406],
                             std=[0.229,0.224,0.225])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),  # converts to [0,1]
                transforms.Normalize(mean=[0.485,0.456,0.406],
                             std=[0.229,0.224,0.225])
            ])
        
    def __len__(self):
        return len(self.matched_df)
    
    def _get_patient_imgs(self, row):
        
        def read_image(path):
            img = Image.open(path).convert("RGB")  # ensure 3 channels
            return img
        
        def get_diameter_from_filename(filename):
            diameter = int(filename.split("_label_")[0].split("diameter_")[1])
            return diameter

        patient_id = row['patient_id']
        img_paths_base = list(glob(os.path.join(self.data_dir, f"{patient_id}_0_0000", "*.png")))
        img_paths_followup = list(glob(os.path.join(self.data_dir, f"{patient_id}_1_0000", "*.png")))
        
        patient_img_output = {"base": [], "followup": [], "base_diameters": [], "followup_diameters": []}
        for p in img_paths_base:
            img = read_image(p)
            patient_img_output["base"].append(img)
            diameter = get_diameter_from_filename(p)
            patient_img_output["base_diameters"].append(diameter)
        
        for p in img_paths_followup:
            img = read_image(p)
            patient_img_output["followup"].append(img)
            diameter = get_diameter_from_filename(p)
            patient_img_output["followup_diameters"].append(diameter)

        patient_img_output["patient_id"] = row["patient_id"]
        patient_img_output["targets"] = {
            "early_recurrence": row["early_recurrence"],
            "overall_survival_24m": row["overall_survival_24m"]
        }
        patient_img_output["demographic_info"] = row["demographic_info"]
        return patient_img_output
    
    def load_data_in_memory(self):
        self._cached_data = []
        for idx in tqdm(range(len(self.matched_df)), desc="Preloading patient images into memory"):
            self._cached_data.append(self._get_patient_imgs(self.matched_df.iloc[idx]))

    def __getitem__(self, idx):
        cached = self._cached_data[idx]  # contains PIL images or numpy arrays

        base = [self.transform(img) for img in cached["base"]]
        followup = [self.transform(img) for img in cached["followup"]]

        return {
            "patient_id": cached["patient_id"],
            "base": base,
            "followup": followup,
            "base_diameters": cached["base_diameters"],
            "followup_diameters": cached["followup_diameters"],
            "targets": cached["targets"],
            "demographic_info": cached["demographic_info"],
        }
        


def custom_collate_fn(batch):
    collated_batch = {
        "patient_ids": [],
        "base":{"img": [], "diameters": [], "batch_idxes": []},
        "followup":{"img": [], "diameters": [], "batch_idxes": []},
        "targets": {
            "early_recurrence": [],
            "overall_survival_24m": []
        },
        "demographic_info": []
    }

    for i, item in enumerate(batch):
        collated_batch["patient_ids"].extend([item["patient_id"]])
        collated_batch["base"]["img"].extend(item["base"])
        collated_batch["followup"]["img"].extend(item["followup"])
        collated_batch["base"]["diameters"].extend(item["base_diameters"])
        collated_batch["followup"]["diameters"].extend(item["followup_diameters"])
        collated_batch["base"]["batch_idxes"].extend([i] * len(item["base"]))
        collated_batch["followup"]["batch_idxes"].extend([i] * len(item["followup"]))
        collated_batch["targets"]["early_recurrence"].append(item["targets"]["early_recurrence"])
        collated_batch["targets"]["overall_survival_24m"].append(item["targets"]["overall_survival_24m"])
        collated_batch["demographic_info"].append(item["demographic_info"])
    
    # Stack images into tensors
    collated_batch["base"]["img"] = torch.stack(collated_batch["base"]["img"])
    collated_batch["followup"]["img"] = torch.stack(collated_batch["followup"]["img"])
    collated_batch["base"]["diameters"] = torch.tensor(collated_batch["base"]["diameters"], dtype=torch.float32)
    collated_batch["followup"]["diameters"] = torch.tensor(collated_batch["followup"]["diameters"], dtype=torch.float32)
    collated_batch["base"]["batch_idxes"] = torch.tensor(collated_batch["base"]["batch_idxes"], dtype=torch.long)
    collated_batch["followup"]["batch_idxes"] = torch.tensor(collated_batch["followup"]["batch_idxes"], dtype=torch.long)
    collated_batch["targets"]["early_recurrence"] = torch.tensor(collated_batch["targets"]["early_recurrence"], dtype=torch.float32)
    collated_batch["targets"]["overall_survival_24m"] = torch.tensor(collated_batch["targets"]["overall_survival_24m"], dtype=torch.float32)
    collated_batch["demographic_info"] = torch.tensor(collated_batch["demographic_info"], dtype=torch.float32)

    return collated_batch

def print_label_statistics(prepared_dataset_df):
    print("Label distribution:")
    print(prepared_dataset_df["early_recurrence"].value_counts())
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
    dataset_train = GooglenetDataset(matched_train_df, preprocessed_data_base_dir, train=True)
    dataset_val = GooglenetDataset(matched_val_df, preprocessed_data_base_dir, train=False)
    dataset_test = GooglenetDataset(matched_test_df, preprocessed_data_base_dir, train=False)
    
    train_loader = DataLoader(dataset_train, batch_size=dataloader_config["batch_size"], shuffle=True, num_workers=6, collate_fn=custom_collate_fn)
    val_loader = DataLoader(dataset_val, batch_size=dataloader_config["batch_size"], shuffle=False, num_workers=2, collate_fn=custom_collate_fn)
    test_loader = DataLoader(dataset_test, batch_size=dataloader_config["batch_size"], shuffle=False, num_workers=2, collate_fn=custom_collate_fn)
    return train_loader, val_loader, test_loader
    

def save_image_grid(tensor_batch, save_path, nrow=4, max_images=16):
    """
    tensor_batch: (N, C, H, W) normalized tensor
    """
    # Take first max_images
    tensor_batch = tensor_batch[:max_images]

    # Undo ImageNet normalization
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1)
    tensor_batch = tensor_batch * std + mean
    tensor_batch = torch.clamp(tensor_batch, 0, 1)

    grid = make_grid(tensor_batch, nrow=nrow)
    # make dir
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    # Convert to PIL
    grid = (grid.permute(1,2,0).cpu().numpy() * 255).astype("uint8")
    Image.fromarray(grid).save(save_path)


def fn_test_loader(loader):
    print(f"Train dataset: {len(loader.dataset)} patients.")
    sample_data = next(iter(loader))
    # Save grids
    save_image_grid(sample_data["base_img"], "googlenet_test/base_grid.png", nrow=4)
    save_image_grid(sample_data["followup_img"], "googlenet_test/followup_grid.png", nrow=4)

    print(f"Example batch keys: {list(sample_data.keys())}")
    print(f"Example base features shape: {sample_data['base_img'].shape}")
    print(f"Example follow-up features shape: {sample_data['followup_img'].shape}")
    print(f"Example early recurrence targets: {sample_data['targets']['early_recurrence']}")
    print(f"Example overall survival 24m targets: {sample_data['targets']['overall_survival_24m']}")
    print(f"Example demographic info: {sample_data['demographic_info']}")
    print(f"Example patient IDs in batch: {sample_data['patient_ids']}")
    print(f"Example base batch indices: {sample_data['base_batch_idxes']}")
    print(f"Example followup batch indices: {sample_data['followup_batch_idxes']}")
    print(f"base diameters in batch: {sample_data['base_diameters']}")
    print(f"followup diameters in batch: {sample_data['followup_diameters']}")

    print(f"len base batch idx {len(sample_data['base_batch_idxes'])}, len followup batch idx {len(sample_data['followup_batch_idxes'])}, len base img {len(sample_data['base_img'])}, len followup img {len(sample_data['followup_img'])}, len base diameters {len(sample_data['base_diameters'])}, len followup diameters {len(sample_data['followup_diameters'])}")
    # for batch in loader:
    #     print(f"Batch base features shape: {batch['base_img'].shape}")
    #     print(f"Batch follow-up features shape: {batch['followup_img'].shape}")
    #     print(f"Batch patient IDs: {batch['patient_ids']}")
    #     print(f"Batch batch indices: {batch['batch_idxes']}")


if __name__ == "__main__":
    data_config_dir = '../configs/data_config.yaml'
    model_config_dir = '../configs/googlenet_config.yaml'
    fold_idx = 0  # Example fold index
    train_loader, val_loader, test_loader = get_cnn_dataloaders(data_config_dir, model_config_dir, fold_idx)
    fn_test_loader(train_loader)