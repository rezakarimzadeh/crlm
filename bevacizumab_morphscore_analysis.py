import argparse
import shutil
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from models.mil import MorphScoreRadiomicsMIL
from models.mlp import MorphScoreStatisticalPoolingMLP
from dataloaders.radiomics_shape_dataloader import get_radiomics_shape_dataloaders
from utils import compute_classification_metrics, save_json, read_yaml
from pathlib import Path
import os
import pandas as pd
import argparse
import numpy as np
import time
import torch
# pl.seed_everything(42)


def get_model_class(model_name: str):
    if model_name == "MorphScoreRadiomicsMIL":
        return MorphScoreRadiomicsMIL
    elif model_name == "MorphScoreStatisticalPoolingMLP":
        return MorphScoreStatisticalPoolingMLP
    else:
        raise ValueError(f"Unknown model name: {model_name}")

def MR_test_model(model, test_loader, output=None):
        model.eval()
        if output is None:
            output = {"y_true": [], "y_pred": [], "y_prob": [], 
                      "base": {"y_true": [], "y_pred": [], "y_prob": []}, 
                      "followup": {"y_true": [], "y_pred": [], "y_prob": []}}
        
        with torch.no_grad():
            for batch in test_loader:
                base_logits, followup_logits = model(batch)
                bevazumab_mask = batch['bevacizumab'] > 0  # Assuming 1 indicates bevacizumab treatment
                # print(f"logits: {base_logits.shape}, Bevacizumab-treated : {bevazumab_mask.shape}")
                logits = torch.cat([base_logits[bevazumab_mask], followup_logits[bevazumab_mask]], dim=0)
                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(logits, dim=1)
                
                base_targets = batch['targets']['morph_score_base']
                followup_targets = batch['targets']['morph_score_followup']
                # Filter to only include bevacizumab-treated patients
                base_logits = base_logits[bevazumab_mask]
                followup_logits = followup_logits[bevazumab_mask]
                base_targets = base_targets[bevazumab_mask]
                followup_targets = followup_targets[bevazumab_mask]
                

                targets = torch.cat([base_targets, followup_targets], dim=0)

                output["y_pred"].extend(preds.cpu().numpy())
                output["y_prob"].extend(probs.cpu().numpy())
                output["y_true"].extend(targets.cpu().numpy())
                
                output["base"]["y_true"].extend(base_targets.cpu().numpy())
                output["base"]["y_pred"].extend(torch.argmax(base_logits, dim=1).cpu().numpy())
                output["base"]["y_prob"].extend(torch.softmax(base_logits, dim=1).cpu().numpy())
                
                output["followup"]["y_true"].extend(followup_targets.cpu().numpy())
                output["followup"]["y_pred"].extend(torch.argmax(followup_logits, dim=1).cpu().numpy())
                output["followup"]["y_prob"].extend(torch.softmax(followup_logits, dim=1).cpu().numpy())

        return output

def mil_mlp(model_name):
    data_config_dir = "./configs/data_config.yaml"
    model_config_dir = "./configs/radiomics_shape_model_config.yaml"
    model_config = read_yaml(model_config_dir)
    feature_to_include = ['shape', 'boundary', 'intensity', 'texture']  
    
    output = None
    for fold_index in range(5):
        _, _, test_loader = get_radiomics_shape_dataloaders(data_config_dir, model_config_dir, feature_to_include, fold_index)

        # define input dimension
        sample_batch = next(iter(test_loader))
        input_dim = sample_batch['base']['features'].shape[-1]
        MODEL_CLASS = get_model_class(model_name)

        str_included_features = "_".join(feature_to_include)
        log_name = f"{model_name}/{str_included_features}"
        save_dir = Path("Results") / log_name  / f"fold_{fold_index}"/"checkpoints"

        #  Test
        best_model_path = save_dir / "best.ckpt"
        best_model = MODEL_CLASS.load_from_checkpoint(best_model_path, config_dir=model_config_dir, features_dim=input_dim, demographic_dim=sample_batch['demographic_info'].shape[-1])
        output = MR_test_model(best_model, test_loader, output)

    classification_metrics = compute_classification_metrics(output)


    save_json(Path("Results") / log_name  / f"bevacizumab_morphscore.json", classification_metrics)


def googlenet():
    from models.googlenet_lstm import MorphScoreGooglenetLSTM
    from dataloaders.googlenet_dataloader import get_cnn_dataloaders

    data_config_dir = "./configs/data_config.yaml"
    model_config_dir = "./configs/googlenet_config.yaml"

    model_config = read_yaml(model_config_dir)
    model_name = "MorphScoreGooglenetLSTM"
    output = None
    for fold_index in range(5):
        train_loader, val_loader, test_loader = get_cnn_dataloaders(data_config_dir, model_config_dir, fold_index)

        log_name = f"{model_name}"
        save_dir = Path("Results") / log_name / f"fold_{fold_index}"
        best_model_path = save_dir / "checkpoints" / "best.ckpt"
        best_model = MorphScoreGooglenetLSTM.load_from_checkpoint(best_model_path, config_dir=model_config_dir, weights_only=False)
        output = MR_test_model(best_model, test_loader, output)
    classification_metrics = compute_classification_metrics(output)

    save_json(Path("Results") / log_name  / f"bevacizumab_morphscore.json", classification_metrics)


def MR_rpn3d_test_model(model, test_loader, output=None):
    from utils import compute_dice
    model.eval()
    if output is None:
        output = {"y_true": [], "y_pred": [], "y_prob": [], 'base_dice': [], 'followup_dice': [],
                    "base": {"y_true": [], "y_pred": [], "y_prob": []},
                    "followup": {"y_true": [], "y_pred": [], "y_prob": []}
                    }
    with torch.no_grad():
        for batch in test_loader:
            base_logits, followup_logits, base_seg_logits, followup_seg_logits = model(batch)
            bevazumab_mask = batch['bevacizumab'] > 0  # Assuming 1 indicates bevacizumab treatment
            logits = torch.cat([base_logits[bevazumab_mask], followup_logits[bevazumab_mask]], dim=0)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)
            
            base_targets = batch['targets']['morph_score_base'][bevazumab_mask]
            followup_targets = batch['targets']['morph_score_followup'][bevazumab_mask]
            targets = torch.cat([base_targets, followup_targets], dim=0)

            base_seg_gt = batch['base_seg'][bevazumab_mask]
            followup_seg_gt = batch['followup_seg'][bevazumab_mask]
            base_seg_logits = base_seg_logits[bevazumab_mask]
            followup_seg_logits = followup_seg_logits[bevazumab_mask]

            base_dice = compute_dice(base_seg_logits, base_seg_gt)
            followup_dice = compute_dice(followup_seg_logits, followup_seg_gt)

            output["y_pred"].extend(preds.cpu().numpy())
            output["y_prob"].extend(probs.cpu().numpy())
            output["y_true"].extend(targets.cpu().numpy())
            
            output["base"]["y_true"].extend(base_targets.cpu().numpy())
            output["base"]["y_pred"].extend(torch.argmax(base_logits, dim=1).cpu().numpy())
            output["base"]["y_prob"].extend(torch.softmax(base_logits, dim=1).cpu().numpy())
            
            output["followup"]["y_true"].extend(followup_targets.cpu().numpy())
            output["followup"]["y_pred"].extend(torch.argmax(followup_logits, dim=1).cpu().numpy())
            output["followup"]["y_prob"].extend(torch.softmax(followup_logits, dim=1).cpu().numpy())
            output["base_dice"].extend(base_dice)
            output["followup_dice"].extend(followup_dice)
    return output


def rpnet3d():
    from models.RPNet import MorphScoreRPNetLightning
    from dataloaders.multi_task_siamese_dataloader import get_mtl_siamese_dataloaders
    data_config_dir = "./configs/data_config.yaml"
    model_config_dir = "./configs/rpnet3d_config.yaml"

    model_config = read_yaml(model_config_dir)
    model_name = "MorphScoreRPNet3D"
    output = None
    for fold_index in range(5):
        train_loader, val_loader, test_loader = get_mtl_siamese_dataloaders(data_config_dir, model_config_dir, fold_index)

    

        log_name = f"{model_name}"
        save_dir = Path("Results") / log_name / f"fold_{fold_index}"
        best_model_path = save_dir / "checkpoints" / "best.ckpt"
        #  Test
        best_model = MorphScoreRPNetLightning.load_from_checkpoint(best_model_path, config_dir=model_config_dir, weights_only=False)
        output = MR_rpn3d_test_model(best_model, test_loader, output)
    
    classification_metrics = compute_classification_metrics(output)
    save_json(Path("Results") / log_name  / f"bevacizumab_morphscore.json", classification_metrics)





def main():
    # mil_mlp("MorphScoreRadiomicsMIL")
    # mil_mlp("MorphScoreStatisticalPoolingMLP")
    googlenet()
    # rpnet3d()

if __name__ == "__main__":
    main()