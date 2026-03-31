import os
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from captum.attr import IntegratedGradients, LayerGradCam, LayerAttribution

from models.RPNet import MorphScoreRPNetLightning, MorphScoreBaseBranch
from dataloaders.multi_task_siamese_dataloader import get_mtl_siamese_dataloaders


def get_model_and_dataloaders(fold_index):
    model_config_dir = "configs/rpnet3d_config.yaml"
    data_config_dir = "configs/data_config.yaml"

    best_model_path = (
        Path("Results")
        / "MorphScoreRPNet3D"
        / f"fold_{fold_index}"
        / "checkpoints"
        / "best.ckpt"
    )

    pl_model = MorphScoreRPNetLightning.load_from_checkpoint(
        best_model_path,
        config_dir=model_config_dir,
        weights_only=False,
    )

    model = MorphScoreBaseBranch(pl_model)
    model = model.to("cuda")
    model.eval()

    train_loader, _, test_loader = get_mtl_siamese_dataloaders(
        data_config_dir, model_config_dir, fold_index
    )
    return model, test_loader


def _normalize_to_0_1(vol):
    vol = vol.astype(np.float32)
    vmin = vol.min()
    vmax = vol.max()
    vol = vol - vmin
    vol = vol / (vmax - vmin + 1e-8)
    return vol


def save_volume_and_saliency_nifti(
    saliency_maps,
    imgs,
    patient_id,
    target_id,
    img_type,
    gt,
    pred,
    method,
    affine=None,
):
    """
    saliency_maps: iterable of [D, H, W]
    imgs:          iterable of [C, D, H, W] or [D, H, W]
    pred:          [N, num_classes] or [N]
    """
    save_dir = f"xai/{method}/rpnet3d"
    os.makedirs(save_dir, exist_ok=True)

    if affine is None:
        affine = np.eye(4, dtype=np.float32)

    for i, (saliency, img) in enumerate(zip(saliency_maps, imgs)):
        img = np.asarray(img)
        saliency = np.asarray(saliency).astype(np.float32)

        if img.ndim == 4:  # [C, D, H, W]
            img_to_save = img[0].astype(np.float32)
        elif img.ndim == 3:  # [D, H, W]
            img_to_save = img.astype(np.float32)
        else:
            raise ValueError(f"Unsupported image shape: {img.shape}")

        saliency_to_save = _normalize_to_0_1(saliency)

        pred_class = int(np.argmax(pred[i])) if pred.ndim == 2 else int(pred[i])

        base_name = (
            f"{img_type}_{patient_id}_target_{target_id}"
            f"_gt_{gt}_pred_{pred_class}"
        )

        nib.save(
            nib.Nifti1Image(img_to_save, affine),
            os.path.join(save_dir, f"{base_name}_image.nii.gz"),
        )
        nib.save(
            nib.Nifti1Image(saliency_to_save, affine),
            os.path.join(save_dir, f"{base_name}_saliency.nii.gz"),
        )


def perform_gradcam(model, test_loader, target_id):
    """
    3D Grad-CAM using Captum LayerGradCam.
    """
    model.eval()

    target_layer = model.sf_conv2.conv
    gradcam = LayerGradCam(model, target_layer)

    input_batch = next(iter(test_loader))
    base_img_tensor = input_batch["base_img"]          # [B, C, D, H, W]
    followup_img_tensor = input_batch["followup_img"]  # [B, C, D, H, W]

    morph_score_base = input_batch["targets"]["morph_score_base"]
    morph_score_followup = input_batch["targets"]["morph_score_followup"]
    patient_ids = input_batch["patient_ids"]

    with torch.no_grad():
        base_predictions = torch.softmax(model(base_img_tensor.to("cuda")), dim=1)
        followup_predictions = torch.softmax(model(followup_img_tensor.to("cuda")), dim=1)

    base_cam = []
    followup_cam = []

    for batch_idx in range(base_img_tensor.shape[0]):
        x = base_img_tensor[batch_idx:batch_idx + 1].to("cuda")  # [1, C, D, H, W]

        attr = gradcam.attribute(x, target=target_id)
        attr = LayerAttribution.interpolate(attr, x.shape[2:])  # (D, H, W)
        attr = attr.squeeze(0).squeeze(0).detach().cpu().numpy()  # [D, H, W]
        attr = np.maximum(attr, 0)
        attr = _normalize_to_0_1(attr)
        base_cam.append(attr)

    for batch_idx in range(followup_img_tensor.shape[0]):
        x = followup_img_tensor[batch_idx:batch_idx + 1].to("cuda")  # [1, C, D, H, W]

        attr = gradcam.attribute(x, target=target_id)
        attr = LayerAttribution.interpolate(attr, x.shape[2:])
        attr = attr.squeeze(0).squeeze(0).detach().cpu().numpy()  # [D, H, W]
        attr = np.maximum(attr, 0)
        attr = _normalize_to_0_1(attr)
        followup_cam.append(attr)

    base_cam = np.stack(base_cam, axis=0)          # [B, D, H, W]
    followup_cam = np.stack(followup_cam, axis=0)  # [B, D, H, W]

    base_imgs_np = base_img_tensor.cpu().numpy()
    followup_imgs_np = followup_img_tensor.cpu().numpy()
    base_preds_np = base_predictions.detach().cpu().numpy()
    followup_preds_np = followup_predictions.detach().cpu().numpy()

    for i, patient_id in enumerate(patient_ids):
        morph_score_base_patient = (
            morph_score_base[i].item()
            if torch.is_tensor(morph_score_base[i])
            else morph_score_base[i]
        )
        morph_score_followup_patient = (
            morph_score_followup[i].item()
            if torch.is_tensor(morph_score_followup[i])
            else morph_score_followup[i]
        )

        save_volume_and_saliency_nifti(
            [base_cam[i]],
            [base_imgs_np[i]],
            patient_id,
            target_id,
            "base",
            gt=morph_score_base_patient,
            pred=np.expand_dims(base_preds_np[i], axis=0),
            method="gradcam",
        )

        save_volume_and_saliency_nifti(
            [followup_cam[i]],
            [followup_imgs_np[i]],
            patient_id,
            target_id,
            "followup",
            gt=morph_score_followup_patient,
            pred=np.expand_dims(followup_preds_np[i], axis=0),
            method="gradcam",
        )


def perform_integrated_gradients(model, test_loader, target_id):
    """
    Assumes 3D input:
      [B, C, D, H, W]
    Output saved after channel reduction:
      [D, H, W]
    """
    model.eval()
    ig = IntegratedGradients(model)

    input_batch = next(iter(test_loader))
    base_img_tensor = input_batch["base_img"]          # [B, C, D, H, W]
    followup_img_tensor = input_batch["followup_img"]  # [B, C, D, H, W]

    morph_score_base = input_batch["targets"]["morph_score_base"]
    morph_score_followup = input_batch["targets"]["morph_score_followup"]
    patient_ids = input_batch["patient_ids"]

    with torch.no_grad():
        base_predictions = torch.softmax(model(base_img_tensor.to("cuda")), dim=1)
        followup_predictions = torch.softmax(model(followup_img_tensor.to("cuda")), dim=1)

    base_attr = []
    followup_attr = []

    for batch_idx in range(base_img_tensor.shape[0]):
        x = base_img_tensor[batch_idx:batch_idx + 1].to("cuda")
        baseline = torch.zeros_like(x)

        attr, _ = ig.attribute(
            x,
            baselines=baseline,
            target=target_id,
            n_steps=40,
            method="gausslegendre",
            return_convergence_delta=True,
            internal_batch_size=1,
        )

        attr = attr.abs().sum(dim=1).detach().cpu().numpy()[0]  # [D, H, W]
        attr = _normalize_to_0_1(attr)
        base_attr.append(attr)

    for batch_idx in range(followup_img_tensor.shape[0]):
        x = followup_img_tensor[batch_idx:batch_idx + 1].to("cuda")
        baseline = torch.zeros_like(x)

        attr, _ = ig.attribute(
            x,
            baselines=baseline,
            target=target_id,
            n_steps=40,
            method="gausslegendre",
            return_convergence_delta=True,
            internal_batch_size=1,
        )

        attr = attr.abs().sum(dim=1).detach().cpu().numpy()[0]  # [D, H, W]
        attr = _normalize_to_0_1(attr)
        followup_attr.append(attr)

    base_attr = np.stack(base_attr, axis=0)          # [B, D, H, W]
    followup_attr = np.stack(followup_attr, axis=0)  # [B, D, H, W]

    base_imgs_np = base_img_tensor.cpu().numpy()
    followup_imgs_np = followup_img_tensor.cpu().numpy()
    base_preds_np = base_predictions.detach().cpu().numpy()
    followup_preds_np = followup_predictions.detach().cpu().numpy()

    for i, patient_id in enumerate(patient_ids):
        morph_score_base_patient = (
            morph_score_base[i].item()
            if torch.is_tensor(morph_score_base[i])
            else morph_score_base[i]
        )
        morph_score_followup_patient = (
            morph_score_followup[i].item()
            if torch.is_tensor(morph_score_followup[i])
            else morph_score_followup[i]
        )

        save_volume_and_saliency_nifti(
            [base_attr[i]],
            [base_imgs_np[i]],
            patient_id,
            target_id,
            "base",
            gt=morph_score_base_patient,
            pred=np.expand_dims(base_preds_np[i], axis=0),
            method="integrated_gradients",
        )

        save_volume_and_saliency_nifti(
            [followup_attr[i]],
            [followup_imgs_np[i]],
            patient_id,
            target_id,
            "followup",
            gt=morph_score_followup_patient,
            pred=np.expand_dims(followup_preds_np[i], axis=0),
            method="integrated_gradients",
        )


def main():
    model, test_loader = get_model_and_dataloaders(fold_index=2)

    for target_id in [0, 1, 2]:
        perform_gradcam(model, test_loader, target_id=target_id)

    for target_id in [0, 1, 2]:
        perform_integrated_gradients(model, test_loader, target_id=target_id)


if __name__ == "__main__":
    main()