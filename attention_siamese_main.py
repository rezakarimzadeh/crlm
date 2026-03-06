import argparse
import shutil
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from models.AttentionSiamese import AttentionMTLLightning
from dataloaders.multi_task_siamese_dataloader import get_mtl_siamese_dataloaders
from utils import attention_siamese_test_model, mtl_compute_classification_metrics, save_json, read_yaml
from pathlib import Path
import os
import pandas as pd
import argparse
import numpy as np
import time
# pl.seed_everything(42)



def train_dl_model(args, fold_index: int):
    data_config_dir = args.data_config_dir
    model_config_dir = args.model_config_dir

    model_config = read_yaml(model_config_dir)
    model_name = args.model_name
    train_loader, val_loader, test_loader = get_mtl_siamese_dataloaders(data_config_dir, model_config_dir, fold_index)
    # define input dimension
    sample_batch = next(iter(train_loader))
    demographic_dim = sample_batch['demographic_info'].shape[-1]
    model = AttentionMTLLightning(config_dir=model_config_dir, demographic_dim=demographic_dim)
  
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
    patience=10,              # epochs to wait before stopping
    verbose=True,
    mode="min"               # "min" for loss, "max" for accuracy/AUC
    )
    log_name = f"{model_name}/multitask_siamese"

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
            log_every_n_steps=1,
            precision="16-mixed",
            )
    #  Train
    print("================= Training Configuration ================")
    print(f"Fold: {fold_index}, LR: {model_config['lr']}, Max Epochs: {model_config['max_epochs']}")

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    print(f"Best checkpoint: {ckpt.best_model_path}")
    #  Test
    best_model = AttentionMTLLightning.load_from_checkpoint(ckpt.best_model_path, config_dir=model_config_dir, demographic_dim=demographic_dim, weights_only=False)
    test_output = attention_siamese_test_model(best_model, test_loader)
    classification_metrics = mtl_compute_classification_metrics(test_output)

    fold_results = {'fold': fold_index, 
                    "classification_metrics": classification_metrics, 
                    'best_checkpoint': ckpt.best_model_path,
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
            if 'dice' in target_key:
                row[target_key] = float(metrics) if metrics is not None else float("nan")
                continue
            for m_name, m_val in metrics.items():
                row[f"{target_key}_{m_name}"] = float(m_val) if m_val is not None else float("nan")

        kfold_rows.append(row)

    df_kfold = pd.DataFrame(kfold_rows)

    def aggregate(df):
        return {
            f"{c}_mean": float(df[c].mean())
            for c in df.columns
        } | {
            f"{c}_std": float(df[c].std(ddof=1))
            for c in df.columns
        }

    kfold_agg = aggregate(df_kfold)



    save_json(model_save_path_last / "fivefold_aggregated_results.json", kfold_agg)


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate models with 5-fold cross-validation.")
    parser.add_argument("--data_config_dir", type=str, default="./configs/data_config.yaml", help="data config file path.")
    parser.add_argument("--model_config_dir", type=str, default="./configs/attention_siamese_config.yaml", help="model config file path.")
    parser.add_argument("--model_name", type=str, default="AttentionSiamese", choices=["AttentionSiamese"], help="model name to use.")

    args = parser.parse_args()
    fivefold_cv(args)

if __name__ == "__main__":
    main()