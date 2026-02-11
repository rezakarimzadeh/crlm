import argparse
import shutil
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from models.cnn_encoder import CnnConcatenation, CnnSiamese
from dataloaders.cnn_dataloader import get_cnn_dataloaders
from utils import test_model, compute_classification_metrics, save_json, read_yaml
from pathlib import Path
import os
import pandas as pd
import argparse
import numpy as np
import time
# pl.seed_everything(42)


def get_model_class(model_name: str):
    if model_name == "CnnConcatenation":
        return CnnConcatenation
    elif model_name == "CnnSiamese":
        return CnnSiamese
    else:
        raise ValueError(f"Unknown model name: {model_name}")


def train_dl_model(args, fold_index: int):
    data_config_dir = args.data_config_dir
    model_config_dir = args.model_config_dir
    target_key = args.target_key

    model_config = read_yaml(model_config_dir)
    model_name = args.model_name
    train_loader, val_loader, test_loader = get_cnn_dataloaders(data_config_dir, model_config_dir, fold_index)
    # define input dimension
    sample_batch = next(iter(train_loader))
    demographic_dim = sample_batch['demographic_info'].shape[-1]

    MODEL_CLASS = get_model_class(model_name)
    model = MODEL_CLASS(demographic_dim=demographic_dim, config_dir=model_config_dir, target_key=target_key)
  
    ckpt = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        save_top_k=1,         
        save_last=True,       
        filename="best",     
        auto_insert_metric_name=False,
    )
    log_name = f"{model_name}/{target_key}"

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
            log_every_n_steps=1,
            )
    #  Train
    print("================= Training Configuration ================")
    print(f"Fold: {fold_index}, LR: {model_config['lr']}, Max Epochs: {model_config['max_epochs']}")

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    print(f"Best checkpoint: {ckpt.best_model_path}")
    #  Test
    best_model = MODEL_CLASS.load_from_checkpoint(ckpt.best_model_path, demographic_dim=demographic_dim, config_dir=model_config_dir, target_key=target_key, weights_only=False)
    test_output = test_model(best_model, test_loader, target_key)
    classification_metrics = compute_classification_metrics(test_output)

    fold_results = {'fold': fold_index, 
                    "classification_metrics": classification_metrics, 
                    'best_checkpoint': ckpt.best_model_path,
                    **test_output}

    save_json(Path(save_dir) / f"results_fold_{fold_index}.json", fold_results)

    return fold_results, Path("Results") / log_name 


def fivefold_cv(args):
    er_rows = []
    model_save_path_last = None

    for fold_idx in range(5):
        results, model_save_path = train_dl_model(args, fold_idx)
        model_save_path_last = model_save_path

        er_metrics = results["classification_metrics"]
        er_rows.append({k: float(v) for k, v in er_metrics.items()})

    df_er = pd.DataFrame(er_rows)

    def aggregate(df):
        return {
            f"{c}_mean": float(df[c].mean())
            for c in df.columns
        } | {
            f"{c}_std": float(df[c].std(ddof=1))
            for c in df.columns
        }

    er_agg = aggregate(df_er)



    save_json(model_save_path_last / "fivefold_aggregated_results.json", er_agg)


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate models with 5-fold cross-validation.")
    parser.add_argument("--data_config_dir", type=str, default="./configs/data_config.yaml", help="data config file path.")
    parser.add_argument("--model_config_dir", type=str, default="./configs/cnn_config.yaml", help="model config file path.")
    parser.add_argument("--model_name", type=str, default="CnnSiamese", choices=["CnnConcatenation", "CnnSiamese"], help="model name to use.")
    parser.add_argument("--target_key", type=str, default="early_recurrence", choices=["early_recurrence", "overall_survival_24m"], help="target key to use for classification.")

    args = parser.parse_args()
    fivefold_cv(args)

if __name__ == "__main__":
    main()