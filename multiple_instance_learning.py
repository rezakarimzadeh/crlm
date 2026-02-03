import argparse
import shutil
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from models.mil import RadiomicsMIL
from dataloaders.radiomics_shape_dataloader import get_radiomics_shape_dataloaders
from utils import test_model, compute_classification_metrics, save_json
from pathlib import Path
import os
import pandas as pd
impohutil
import argparse
# pl.seed_everything(42)




def train_dl_model(argparse, fold_index: int):
    data_config_dir = './configs/data_config.yaml'
    model_config_dir = './configs/radiomics_shape_model_config.yaml'
    feature_to_include = ['shape', 'boundary', 'firstorder', 'glcm', 'glszm_glrlm']  
    train_loader, val_loader, test_loader = get_radiomics_shape_dataloaders(data_config_dir, model_config_dir, feature_to_include, fold_index)

    # define input dimension
    sample_batch = next(iter(train_loader))
    input_dim = sample_batch['base']['features'].shape[-1]
    
    model = RadiomicsMIL(features_dim=input_dim, demographic_dim=sample_batch['demographic_info'].shape[-1], config_dir=model_config_dir)
    print("================= Training Configuration ================")
    print(f"Input feature dimension: {input_dim}, Model: {argparse.model_name}, Fold: {fold_index}, coords: {argparse.use_coords}, demographic: {argparse.use_demographic}")

    ckpt = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        save_top_k=1,         
        save_last=True,       
        filename="best",     
        auto_insert_metric_name=False,
    )
    str_included_features = "_".join(feature_to_include)
    log_name = f"mil_{str_included_features}"

    save_dir = os.path.join("Results", log_name, f"fold_{fold_index}")
    if os.path.exists(save_dir):
        # delete existing directory
        shutil.rmtree(save_dir)
    #  TensorBoard logger
    tb_logger = TensorBoardLogger(save_dir="Results", name=log_name, version=f"fold_{fold_index}")

    #  Trainer 
    trainer = pl.Trainer(
            max_epochs=model_configs['max_epochs'],
            callbacks=[ckpt],
            logger=tb_logger,
            accelerator="auto",
            devices="auto",
            )
    #  Train
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    print(f"Best checkpoint: {ckpt.best_model_path}")
    #  Test
    best_model = RadiomicsMIL.load_from_checkpoint(ckpt.best_model_path, config_dir=model_config_dir, features_dim=input_dim, demographic_dim=sample_batch['demographic_info'].shape[-1])

    test_output = test_model(best_model, test_loader)
    overall_survival_results = compute_classification_metrics("overall_survival", test_output)
    early_response_results = compute_classification_metrics("early_response", test_output)
    fold_results = {"overall_survival": overall_survival_results, "early_response": early_response_results}
    print(f"Test Results Fold {fold_index}: {classification_results}")
    fold_results = {'fold': fold_index, **classification_results, 'best_checkpoint': ckpt.best_model_path,
                    'used_features': feature_to_include,  
                    'y_true': test_output['y_true'], 'y_pred': test_output['y_pred'], 'y_prob': test_output['y_prob']}

    save_json(Path(save_dir) / f"results_fold_{fold_index}.json", fold_results)

    return pd.DataFrame([fold_results]), Path("Results") / log_name 


def fivefold_cv(argparse):
    aggrigation_list = []
    center2_aggrigation_list = []
    for fold_idx in range(5):
        classification_results, model_save_path = train_dl_model(argparse, fold_idx)
        aggrigation_list.append(classification_results)

    aggregated_metrics = pd.concat(aggrigation_list).groupby("model").agg(
        Accuracy_Mean=("accuracy", "mean"),
        Accuracy_STD=("accuracy", "std"),
        ROC_AUC_Mean=("roc_auc", "mean"),
        ROC_AUC_STD=("roc_auc", "std"),
        Precision_Mean=("precision", "mean"),
        Precision_STD=("precision", "std"),
        Recall_Mean=("recall", "mean"),
        Recall_STD=("recall", "std"),
        Specificity_Mean=("specificity", "mean"),
        Specificity_STD=("specificity", "std"),
        F1_Score_Mean=("f1_score", "mean"),
        F1_Score_STD=("f1_score", "std"),
    ).reset_index()
    print("\nAggregated Model Performance on Masih Dataset over 5 folds:")
    # aggregated_metrics['use_coords'] = argparse.use_coords
    print(aggregated_metrics)
    save_json(model_save_path / "aggregated_results.json", aggregated_metrics.to_dict(orient="records"))


def main():
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("True", "yes", "true", "t", "1"):
            return True
        elif v.lower() in ("False""no", "false", "f", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")

    parser = argparse.ArgumentParser(description="Train and evaluate models with 5-fold cross-validation.")
    parser.add_argument("--data_root", type=str, default="/home/reza/Documents/Reza_projects/08_drarabi_lymphnodes/new_dataset", help="Base directory for the dataset.")
    parser.add_argument("--model_name", type=str, default="mil", choices=["transformer", "deep_sets", "mil", "ml_models", 'graph', 'set_transformer'], help="Name of the model to train.")
    parser.add_argument("--use_coords", type=str2bool, default=False, help="Whether to use coordinates features.")
    parser.add_argument("--use_demographic", type=str2bool, default=False, help="Whether to use demographic features.")
    args = parser.parse_args()
    fivefold_cv(args)

if __name__ == "__main__":
    main()