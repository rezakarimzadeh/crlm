import argparse
import shutil
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from models.mtl_mil import MTLRadiomicsMIL
from models.mlp import StatisticalPoolingMLP
from dataloaders.radiomics_shape_dataloader import get_radiomics_shape_dataloaders
from utils import mtl_test_model, mtl_compute_classification_metrics, save_json, read_yaml
from pathlib import Path
import os
import pandas as pd
import argparse
import numpy as np
import time
# pl.seed_everything(42)


def get_model_class(model_name: str):
    if model_name == "MTLRadiomicsMIL":
        return MTLRadiomicsMIL
    elif model_name == "StatisticalPoolingMLP":
        return StatisticalPoolingMLP
    else:
        raise ValueError(f"Unknown model name: {model_name}")


def train_dl_model(args, fold_index: int):
    data_config_dir = args.data_config_dir
    model_config_dir = args.model_config_dir
    target_keys = args.target_keys
    model_config = read_yaml(model_config_dir)
    feature_to_include = args.feature_to_include  
    model_name = args.model_name
    train_loader, val_loader, test_loader = get_radiomics_shape_dataloaders(data_config_dir, model_config_dir, feature_to_include, fold_index)

    # define input dimension
    sample_batch = next(iter(train_loader))
    input_dim = sample_batch['base']['features'].shape[-1]
    MODEL_CLASS = get_model_class(model_name)
    model = MODEL_CLASS(features_dim=input_dim, demographic_dim=sample_batch['demographic_info'].shape[-1], config_dir=model_config_dir, target_keys=target_keys)
  
    ckpt = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        save_top_k=1,         
        save_last=True,       
        filename="best",     
        auto_insert_metric_name=False,
    )
    early_stop_callback = EarlyStopping(
    monitor="val_loss",      # metric to monitor
    min_delta=0.00,          # minimum change to qualify as improvement
    patience=30,              # epochs to wait before stopping
    verbose=True,
    mode="min"               # "min" for loss, "max" for accuracy/AUC
    )

    str_included_features = "_".join(feature_to_include)
    log_name = f"{model_name}/{'_'.join(target_keys)}/{str_included_features}"

    save_dir = Path("Results") / log_name / f"fold_{fold_index}"
    if save_dir.exists():
        shutil.rmtree(save_dir, ignore_errors=True)
        time.sleep(0.2)
    #  TensorBoard logger
    tb_logger = TensorBoardLogger(save_dir="Results", name=log_name, version=f"fold_{fold_index}")

    #  Trainer 
    trainer = pl.Trainer(
            max_epochs=model_config['max_epochs'],
            callbacks=[ckpt, early_stop_callback],
            logger=tb_logger,
            accelerator="auto",
            devices="auto",
            )
    #  Train
    print("================= Training Configuration ================")
    print(f"Input feature dimension: {input_dim}, Fold: {fold_index}, included features: {args.feature_to_include}, LR: {model_config['lr']}, Max Epochs: {model_config['max_epochs']}")

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    print(f"Best checkpoint: {ckpt.best_model_path}")
    #  Test
    best_model = MODEL_CLASS.load_from_checkpoint(ckpt.best_model_path, config_dir=model_config_dir, features_dim=input_dim, demographic_dim=sample_batch['demographic_info'].shape[-1], target_keys=target_keys)
    test_output = mtl_test_model(best_model, test_loader, target_keys)
    classification_metrics = mtl_compute_classification_metrics(test_output)

    fold_results = {'fold': fold_index, 
                    "classification_metrics": classification_metrics, 
                    'best_checkpoint': ckpt.best_model_path,
                    'used_features': feature_to_include,  
                    **test_output}

    save_json(Path(save_dir) / f"results_fold_{fold_index}.json", fold_results)

    return fold_results, Path("Results") / log_name 


def fivefold_cv(args):
    kfold_rows = []
    model_save_path_last = None

    for fold_idx in range(5):
        results, model_save_path = train_dl_model(args, fold_idx)
        model_save_path_last = model_save_path

        classification_metrics = results["classification_metrics"]

        row = {}
        for target_key, metrics in classification_metrics.items():   # metrics is a dict
            for m_name, m_val in metrics.items():
                row[f"{target_key}_{m_name}"] = float(m_val) if m_val is not None else float("nan")

        kfold_rows.append(row)

    df_kfold = pd.DataFrame(kfold_rows)

    # detect target keys from column prefixes
    target_keys = set(c.split("_")[0] for c in df_kfold.columns)

    kfold_agg = {}

    for target_key in target_keys:
        target_cols = [c for c in df_kfold.columns if c.startswith(f"{target_key}_")]

        agg_dict = {}
        for col in target_cols:
            metric_name = col.replace(f"{target_key}_", "")
            agg_dict[f"{metric_name}_mean"] = float(df_kfold[col].mean())
            agg_dict[f"{metric_name}_std"] = float(df_kfold[col].std(ddof=1))

        kfold_agg[target_key] = agg_dict

    save_json(model_save_path_last / "fivefold_aggregated_results.json", kfold_agg)


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate models with 5-fold cross-validation.")
    parser.add_argument("--data_config_dir", type=str, default="./configs/data_config.yaml", help="data config file path.")
    parser.add_argument("--model_config_dir", type=str, default="./configs/radiomics_shape_model_config.yaml", help="model config file path.")
    parser.add_argument("--feature_to_include", type=str, default=['shape', 'boundary', 'intensity', 'texture'], help="model name to use.")
    parser.add_argument("--model_name", type=str, default="MTLRadiomicsMIL", choices=["MTLRadiomicsMIL", "StatisticalPoolingMLP"], help="model name to use.")
    parser.add_argument("--target_keys", type=list, default=["early_recurrence","overall_survival_24m", "pathology"], choices=["early_recurrence", "overall_survival_24m", "pathology"], help="target key to use for classification.")

    args = parser.parse_args()
    fivefold_cv(args)

if __name__ == "__main__":
    main()