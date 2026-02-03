import yaml
import torch 
import numpy as np
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix


def read_yaml_file(file_path: str) -> dict:
    """Read a YAML configuration file and return its contents as a dictionary."""
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def save_json(file_path: str, data: dict):
    """Save a dictionary to a JSON file."""
    import json
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)

def test_model(model, test_loader):
        model.eval()
        all_preds = {"overall_survival": [], "early_response": []}
        all_probs = {"overall_survival": [], "early_response": []}
        all_labels = {"overall_survival": [], "early_response": []}
        try:
            device = model.device
        except:
             device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        with torch.no_grad():
            for batch in test_loader:
                logits = model(batch)
                survival_logits = logits['survival']
                early_response_logits = logits['early_response']

                survival_probs = torch.softmax(survival_logits, dim=1)[:, 1]
                early_response_probs = torch.softmax(early_response_logits, dim=1)[:, 1]

                survival_preds = torch.argmax(survival_logits, dim=1)
                early_response_preds = torch.argmax(early_response_logits, dim=1)

                all_preds["overall_survival"].extend(survival_preds.cpu().numpy())
                all_probs["overall_survival"].extend(survival_probs.cpu().numpy())
                all_labels["overall_survival"].extend(batch['target']['overall_survival_24m'].cpu().numpy())

                all_preds["early_response"].extend(early_response_preds.cpu().numpy())
                all_probs["early_response"].extend(early_response_probs.cpu().numpy())
                all_labels["early_response"].extend(batch['target']['early_response'].cpu().numpy())
            output = {
                "y_true": all_labels,
                "y_pred": all_preds,
                "y_prob": all_probs
            }
        return output

def compute_classification_metrics(target_name, test_output):
    y_true = np.array(test_output["y_true"][target_name])
    y_pred = np.array(test_output["y_pred"][target_name])
    y_prob = np.array(test_output["y_prob"][target_name])
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_prob)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp)

    metrics = {
        'target': target_name,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'specificity': specificity,
        'f1_score': f1,
        'roc_auc': roc_auc
    }

    return metrics