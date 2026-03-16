import yaml
import torch 
from pathlib import Path
import json
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

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

def test_model(model, test_loader, target_key):
        model.eval()
        output = {"y_true": [], "y_pred": [], "y_prob": []}
        try:
            device = model.device
        except:
             device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        with torch.no_grad():
            for batch in test_loader:
                logits = model(batch)

                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(logits, dim=1)

                output["y_pred"].extend(preds.cpu().numpy())
                output["y_prob"].extend(probs.cpu().numpy())
                output["y_true"].extend(batch['targets'][target_key].cpu().numpy())
        return output

# def compute_classification_metrics(test_output):
#     y_true = np.array(test_output["y_true"])
#     y_pred = np.array(test_output["y_pred"])
#     y_prob = np.array(test_output["y_prob"])
#     accuracy = accuracy_score(y_true, y_pred)
#     precision = precision_score(y_true, y_pred)
#     recall = recall_score(y_true, y_pred)
#     f1 = f1_score(y_true, y_pred)
#     roc_auc = roc_auc_score(y_true, y_prob)

#     tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
#     specificity = tn / (tn + fp)

#     metrics = {
#         'accuracy': accuracy,
#         'precision': precision,
#         'recall': recall,
#         'specificity': specificity,
#         'f1_score': f1,
#         'roc_auc': roc_auc
#     }

#     if 'base_dice' in test_output and 'followup_dice' in test_output:
#         metrics['base_dice'] = np.mean(test_output['base_dice'])
#         metrics['followup_dice'] = np.mean(test_output['followup_dice'])

#     return metrics



def compute_classification_metrics(test_output):
    y_true = np.array(test_output["y_true"])
    y_pred = np.array(test_output["y_pred"])
    y_prob = np.array(test_output["y_prob"])
    # masking to handle -1 targets (missing labels)
    mask = y_true != -1
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    y_prob = y_prob[mask]
    n_classes = y_prob.shape[1] if y_prob.ndim == 2 else 2

    accuracy = accuracy_score(y_true, y_pred)

    if n_classes == 2:
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        # y_prob can be shape (N,) or (N,2)
        if y_prob.ndim == 2:
            roc_auc = roc_auc_score(y_true, y_prob[:, 1])
        else:
            roc_auc = roc_auc_score(y_true, y_prob)

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    else:
        precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
        recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

        roc_auc = roc_auc_score(y_true, y_prob, multi_class="ovr", average="weighted")
        specificity = None  # not directly defined as a single value in multiclass

    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1_score": f1,
        "roc_auc": roc_auc,
    }

    if "base_dice" in test_output and "followup_dice" in test_output:
        metrics["base_dice"] = np.mean(test_output["base_dice"])
        metrics["followup_dice"] = np.mean(test_output["followup_dice"])

    return metrics


def mtl_test_model(model, test_loader, target_keys):
    model.eval()
    outputs = {k: {"y_true": [], "y_pred": [], "y_prob": []} for k in target_keys}

    try:
        device = model.device
    except:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with torch.no_grad():
        for batch in test_loader:
            logits = model(batch)

            for target_key in target_keys:
                preds = torch.argmax(logits[target_key], dim=1)
                gt = batch['targets'][target_key]
                if target_key == "pathology":
                    mask = gt != -1
                    preds = preds[mask]
                    gt = gt[mask]
                    probs = torch.softmax(logits[target_key], dim=1)[mask]
                    outputs[target_key]["y_prob"].extend(probs.cpu().numpy())
                else:
                    # binary: store P(class=1) [B]
                    probs = torch.softmax(logits[target_key], dim=1)[:, 1]
                    outputs[target_key]["y_prob"].extend(probs.cpu().numpy())
                # print(f"Target: {target_key}, GT labels shape: {gt.shape}, Preds shape: {preds.shape}")
                outputs[target_key]["y_pred"].extend(preds.cpu().numpy())
                outputs[target_key]["y_true"].extend(gt.cpu().numpy())

    return outputs


def mtl_compute_classification_metrics(test_outputs):
    results = {}
    for target_key, test_output in test_outputs.items():
        if 'dice' in target_key:
            results[target_key] = np.mean(test_output)  # already computed dice scores
            continue

        y_true = np.array(test_output["y_true"])
        y_pred = np.array(test_output["y_pred"])
        y_prob = np.array(test_output["y_prob"])

        # minimal fix: handle empty targets
        if len(y_true) == 0 or len(y_pred) == 0:
            results[target_key] = {
                'accuracy': np.nan,
                'precision': np.nan,
                'recall': np.nan,
                'specificity': np.nan,
                'f1_score': np.nan,
                'roc_auc': np.nan
            }
            continue

        if target_key == "pathology":
            accuracy = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
            C = y_prob.shape[1]
            roc_auc = roc_auc_score(y_true, y_prob, multi_class='ovr', labels=np.arange(C))
            specificity = np.nan
        else:
            accuracy = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            roc_auc = roc_auc_score(y_true, y_prob)

            tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        results[target_key] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'specificity': specificity,
            'f1_score': f1,
            'roc_auc': roc_auc
        }

    return results

# RpNet3D-specific test function that calls the generic mtl test and metric functions

def compute_dice(pred_probs, gt_masks, threshold=0.5, logits=False):
    device = pred_probs.device
    gt_masks = gt_masks.to(device)
    # pred probs: [B, 1, D, H, W], gt_masks: [B, 1, D, H, W]
    if logits:
        pred_probs = torch.sigmoid(pred_probs)
    pred_masks = (pred_probs > threshold).float()
    gt_masks = gt_masks.float()
    intersection = (pred_masks * gt_masks).sum(dim=[1, 2, 3, 4])
    union = pred_masks.sum(dim=[1, 2, 3, 4]) + gt_masks.sum(dim=[1, 2, 3, 4])
    dice = (2. * intersection + 1e-6) / (union + 1e-6)
    return dice.cpu().numpy().tolist()

def rpn3d_test_model(model, test_loader, target_key):
        model.eval()
        output = {"y_true": [], "y_pred": [], "y_prob": [], 'base_dice': [], 'followup_dice': []}
        try:
            device = model.device
        except:
             device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        with torch.no_grad():
            for batch in test_loader:
                logits, base_seg_logits, followup_seg_logits = model(batch)
                
                base_seg_gt = batch['base_seg']
                followup_seg_gt = batch['followup_seg']


                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(logits, dim=1)

                output["y_pred"].extend(preds.cpu().numpy())
                output["y_prob"].extend(probs.cpu().numpy())
                output["y_true"].extend(batch['targets'][target_key].cpu().numpy())
                output["base_dice"].extend(compute_dice(base_seg_logits, base_seg_gt))
                output["followup_dice"].extend(compute_dice(followup_seg_logits, followup_seg_gt))
        return output


def attention_siamese_test_model(model, test_loader):
    model.eval()
    target_keys = ["early_recurrence", "overall_survival_24m", "pathology"]
    outputs = {k: {"y_true": [], "y_pred": [], "y_prob": []} for k in target_keys}
    outputs['pre_segmentation_dice'] = []
    outputs['post_segmentation_dice'] = []
    try:
        device = model.device
    except:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with torch.no_grad():
        for batch in test_loader:
            logits = model(batch)

            for target_key in batch['targets'].keys():
                preds = torch.argmax(logits[f"classifier_logits_{target_key}"], dim=1)
                gt = batch['targets'][target_key]
                if target_key == "pathology":
                    mask = gt != -1
                    preds = preds[mask]
                    gt = gt[mask]
                    probs = torch.softmax(logits[f"classifier_logits_{target_key}"], dim=1)[mask]
                    outputs[target_key]["y_prob"].extend(probs.cpu().numpy())
                else:
                    # binary: store P(class=1) [B]
                    probs = torch.softmax(logits[f"classifier_logits_{target_key}"], dim=1)[:, 1]
                    outputs[target_key]["y_prob"].extend(probs.cpu().numpy())
                # print(f"Target: {target_key}, GT labels shape: {gt.shape}, Preds shape: {preds.shape}")
                outputs[target_key]["y_pred"].extend(preds.cpu().numpy())
                outputs[target_key]["y_true"].extend(gt.cpu().numpy())
            # compute segmentation dice
            base_seg_logits = logits['pre_seg_logits']
            followup_seg_logits = logits['post_seg_logits']
            base_seg_gt = batch['base_seg']
            followup_seg_gt = batch['followup_seg']
            outputs['pre_segmentation_dice'].extend(compute_dice(base_seg_logits, base_seg_gt, logits=True))
            outputs['post_segmentation_dice'].extend(compute_dice(followup_seg_logits, followup_seg_gt, logits=True))
    return outputs