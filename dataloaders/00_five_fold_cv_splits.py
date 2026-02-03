import os
from pathlib import Path
from sklearn.model_selection import KFold, train_test_split, StratifiedKFold, StratifiedShuffleSplit
from utils import load_json, save_to_json, read_yaml


def split_data_5foldCV(data_root, output_dir, seed=11):
    all_data = load_json(Path(data_root) / "img_groups.json")
    print(f"Total samples: {len(all_data)}")
    kf = KFold(n_splits=5, random_state=seed, shuffle=True)
    for i, (train_index, test_index) in enumerate(kf.split(all_data)):
        train_val_data = [all_data[idx] for idx in train_index]
        test_data = [all_data[idx] for idx in test_index]
        train_data, val_data = train_test_split(
            train_val_data, test_size=0.2, random_state=seed
        )
        split_dict = {
            "train": train_data,
            "val": val_data,
            "test": test_data
        }
        print(f"Fold {i+1}: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")
        split_filename = os.path.join(output_dir, f"five_fold_cv_split_{i}.json")
        os.makedirs(os.path.dirname(split_filename), exist_ok=True)
        save_to_json(split_dict, split_filename)
        print(f"Saved fold {i+1} split to {split_filename}")


if __name__ == "__main__":
    config_dir = "../configs/data_config.yaml"
    data_config = read_yaml(config_dir)
    data_root = data_config["preprocessed_data_base_dir"]
    output_dir = Path(data_config["preprocessed_data_base_dir"]) / "five_fold_cv_splits"
    os.makedirs(output_dir, exist_ok=True)
    split_data_5foldCV(data_root, output_dir)