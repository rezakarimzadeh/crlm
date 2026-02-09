import yaml
import torch 
import numpy as np
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from pathlib import Path
import json

def read_yaml(file_path: str) -> dict:
    """Read a YAML configuration file and return its contents as a dictionary."""
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def _to_jsonable(x):
    # torch tensors
    try:
        import torch
        if torch.is_tensor(x):
            return x.detach().cpu().tolist()
    except Exception:
        pass

    # numpy
    try:
        import numpy as np
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, (np.integer,)):
            return int(x)
        if isinstance(x, (np.floating,)):
            return float(x)
        if isinstance(x, (np.bool_,)):
            return bool(x)
    except Exception:
        pass

    # python containers
    if isinstance(x, dict):
        return {k: _to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_to_jsonable(v) for v in x]

    # pathlib
    if isinstance(x, Path):
        return str(x)

    return x

def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(_to_jsonable(data), f, indent=4)

def read_json(file_path: str) -> dict:
    """Read a JSON file and return its contents as a dictionary."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def test_model(model, test_loader):
        model.eval()
        output = {"early_response": {"y_true": [], "y_pred": [], "y_prob": []}}
        try:
            device = model.device
        except:
             device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        with torch.no_grad():
            for batch in test_loader:
                logits = model(batch)
                # survival_logits = logits['survival']
                early_response_logits = logits['early_response']

                # survival_probs = torch.softmax(survival_logits, dim=1)[:, 1]
                early_response_probs = torch.softmax(early_response_logits, dim=1)[:, 1]

                # survival_preds = torch.argmax(survival_logits, dim=1)
                early_response_preds = torch.argmax(early_response_logits, dim=1)

                # output["overall_survival"]["y_pred"].extend(survival_preds.cpu().numpy())
                # output["overall_survival"]["y_prob"].extend(survival_probs.cpu().numpy())
                # output["overall_survival"]["y_true"].extend(batch['targets']['overall_survival_24m'].cpu().numpy())

                output["early_response"]["y_pred"].extend(early_response_preds.cpu().numpy())
                output["early_response"]["y_prob"].extend(early_response_probs.cpu().numpy())
                output["early_response"]["y_true"].extend(batch['targets']['early_response'].cpu().numpy())
        return output

def compute_classification_metrics(target_name, test_output):
    y_true = np.array(test_output[target_name]["y_true"])
    y_pred = np.array(test_output[target_name]["y_pred"])
    y_prob = np.array(test_output[target_name]["y_prob"])
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_prob)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp)

    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'specificity': specificity,
        'f1_score': f1,
        'roc_auc': roc_auc
    }

    return metrics

