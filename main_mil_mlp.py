import argparse
import shutil
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from models.mil import RadiomicsMIL
from models.mlp import StatisticalPoolingMLP
from dataloaders.radiomics_shape_dataloader import get_radiomics_shape_dataloaders
from utils import test_model, compute_classification_metrics, save_json, read_yaml
from pathlib import Path
import os
import pandas as pd
import argparse
import numpy as np
import time
# pl.seed_everything(42)


def get_model_class(model_name: str):
    if model_name == "RadiomicsMIL":
        return RadiomicsMIL
    elif model_name == "StatisticalPoolingMLP":
        return StatisticalPoolingMLP
    else:
        raise ValueError(f"Unknown model name: {model_name}")


def train_dl_model(args, fold_index: int):
    data_config_dir = args.data_config_dir
    model_config_dir = args.model_config_dir
    model_config = read_yaml(model_config_dir)
    feature_to_include = args.feature_to_include  
    model_name = args.model_name
    train_loader, val_loader, test_loader = get_radiomics_shape_dataloaders(data_config_dir, model_config_dir, feature_to_include, fold_index)

    # define input dimension
    sample_batch = next(iter(train_loader))
    input_dim = sample_batch['base']['features'].shape[-1]
    MODEL_CLASS = get_model_class(model_name)
    model = MODEL_CLASS(features_dim=input_dim, demographic_dim=sample_batch['demographic_info'].shape[-1], config_dir=model_config_dir)
  
    ckpt = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        save_top_k=1,         
        save_last=True,       
        filename="best",     
        auto_insert_metric_name=False,
    )
    str_included_features = "_".join(feature_to_include)
    log_name = f"{model_name}_{str_included_features}"

    save_dir = Path("Results") / log_name / f"fold_{fold_index}"
    if save_dir.exists():
        shutil.rmtree(save_dir, ignore_errors=True)
        time.sleep(0.2)
    #  TensorBoard logger
    tb_logger = TensorBoardLogger(save_dir="Results", name=log_name, version=f"fold_{fold_index}")

    #  Trainer 
    trainer = pl.Trainer(
            max_epochs=model_config['max_epochs'],
            callbacks=[ckpt],
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
    best_model = MODEL_CLASS.load_from_checkpoint(ckpt.best_model_path, config_dir=model_config_dir, features_dim=input_dim, demographic_dim=sample_batch['demographic_info'].shape[-1])
    test_output = test_model(best_model, test_loader)
    # overall_survival_classification_metrics = compute_classification_metrics("overall_survival", test_output)
    early_response_classification_metrics = compute_classification_metrics("early_response", test_output)
    # fold_results = {"overall_survival": overall_survival_classification_metrics, "early_response": early_response_classification_metrics}
    fold_results = {"early_response": early_response_classification_metrics}

    fold_results = {'fold': fold_index, 
                    # "overall_survival_classification_metrics": overall_survival_classification_metrics, 
                    "early_response_classification_metrics": early_response_classification_metrics, 
                    'best_checkpoint': ckpt.best_model_path,
                    'used_features': feature_to_include,  
                    **test_output}

    save_json(Path(save_dir) / f"results_fold_{fold_index}.json", fold_results)

    return fold_results, Path("Results") / log_name 


def fivefold_cv(args):
    os_rows = []
    er_rows = []
    model_save_path_last = None

    for fold_idx in range(5):
        results, model_save_path = train_dl_model(args, fold_idx)
        model_save_path_last = model_save_path

        # os_metrics = results["overall_survival_classification_metrics"]
        er_metrics = results["early_response_classification_metrics"]
        # convert numpy scalars to python floats
        # os_rows.append({k: float(v) for k, v in os_metrics.items()})
        er_rows.append({k: float(v) for k, v in er_metrics.items()})

    df_os = pd.DataFrame(os_rows)
    df_er = pd.DataFrame(er_rows)

    def aggregate(df):
        return {
            f"{c}_mean": float(df[c].mean())
            for c in df.columns
        } | {
            f"{c}_std": float(df[c].std(ddof=1))
            for c in df.columns
        }

    os_agg = aggregate(df_os)
    er_agg = aggregate(df_er)

    out = {
        "overall_survival": os_agg,
        "early_response": er_agg,
    }

    save_json(model_save_path_last / "fivefold_aggregated_results.json", out)


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate models with 5-fold cross-validation.")
    parser.add_argument("--data_config_dir", type=str, default="./configs/data_config.yaml", help="data config file path.")
    parser.add_argument("--model_config_dir", type=str, default="./configs/radiomics_shape_model_config.yaml", help="model config file path.")
    parser.add_argument("--feature_to_include", type=str, default=['shape', 'boundary', 'intensity', 'texture'], help="model name to use.")
    parser.add_argument("--model_name", type=str, default="RadiomicsMIL", choices=["RadiomicsMIL", "StatisticalPoolingMLP"], help="model name to use.")

    args = parser.parse_args()
    fivefold_cv(args)

if __name__ == "__main__":
    main()