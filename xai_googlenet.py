import math

from models.googlenet_lstm import MorphScoreGooglenetLSTM, MorphScoreBaseBranch
from dataloaders.googlenet_dataloader import get_cnn_dataloaders
from pathlib import Path
import matplotlib.pyplot as plt
from captum.attr import IntegratedGradients
import torch
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import numpy as np
import os

def get_model_and_dataloaders(fold_index):

    model_config_dir = "configs/googlenet_config.yaml"
    data_config_dir = "configs/data_config.yaml"

    best_model_path = Path("Results") / "MorphScoreGooglenetLSTM" / f"fold_{fold_index}" / "checkpoints" / "best.ckpt"
    pl_model = MorphScoreGooglenetLSTM.load_from_checkpoint(best_model_path, config_dir=model_config_dir, weights_only=False)
    model = MorphScoreBaseBranch(pl_model)
    model.eval()

    train_loader, _, test_loader = get_cnn_dataloaders(data_config_dir, model_config_dir, fold_index)
    return model, test_loader



def plt_save_cam_on_image(cam_maps, rgb_imgs, patient_id, target_id, img_type, gt, pred, method, img_weight=0.9):
    n_imgs = len(cam_maps)
    ncols = min(3, n_imgs)
    nrows = math.ceil(n_imgs / ncols)

    fig_w = 3 * ncols
    fig_h = 3 * nrows
    fig = plt.figure(figsize=(fig_w, fig_h))

    for i, (cam_map, rgb_img) in enumerate(zip(cam_maps, rgb_imgs)):
        rgb_img = np.transpose(rgb_img, (1, 2, 0))  # [H, W, C]
        rgb_img = (rgb_img - rgb_img.min()) / (rgb_img.max() - rgb_img.min() + 1e-8)  # normalize to [0,1]
        visualization = show_cam_on_image(rgb_img, cam_map, use_rgb=True, image_weight=img_weight)

        ax = plt.subplot(nrows, ncols, i + 1)

        # estimate subplot width in inches and scale title fontsize from it
        subplot_width_in = fig_w / ncols
        title_fs = min(12, max(6, subplot_width_in * 3))

        ax.set_title(f"Pred: {pred[i]}", fontsize=title_fs, wrap=True)
        ax.imshow(visualization)
        ax.axis('off')

    plt.suptitle(f"Patient {patient_id}, target {target_id}, GT: {gt}", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    os.makedirs(f"xai/{method}/googlenet_lstm", exist_ok=True)
    plt.savefig(f"xai/{method}/googlenet_lstm/{img_type}_{patient_id}_target_{target_id}.png", dpi=200)
    plt.close()

def perform_gradcam(model, test_loader, target_id):
    model.eval()
    target_layers = [model.cnn.model.inception5b]
    targets = [ClassifierOutputTarget(target_id)]


    input_batch = next(iter(test_loader))
    base_img_tensor = input_batch['base']['img']  # shape [B, C, H, W]
    followup_img_tensor = input_batch['followup']['img']  # shape [B, C, H, W]
    
    morph_score_base = input_batch['targets']['morph_score_base']  # shape [B]
    morph_score_followup = input_batch['targets']['morph_score_followup']  # shape [B]
    with torch.no_grad():
        base_predictions = torch.softmax(model(base_img_tensor.to("cuda")), dim=1)  # shape [B, num_classes]
        followup_predicitons = torch.softmax(model(followup_img_tensor.to("cuda")), dim=1)  # shape [B, num_classes]

    base_grayscale_cam = []
    followup_grayscale_cam = []
    for batch_idx in range(base_img_tensor.shape[0]):
        single_base_img_tensor = base_img_tensor[batch_idx:batch_idx+1].to("cuda")  # shape [1, C, H, W]
        with GradCAMPlusPlus(model=model, target_layers=target_layers) as cam:
            single_base_grayscale_cam = cam(
                input_tensor=single_base_img_tensor,
                targets=targets,
                aug_smooth=False,
                eigen_smooth=False,
            )
        base_grayscale_cam.append(single_base_grayscale_cam[0])  # shape [H, W]
    
    for batch_idx in range(followup_img_tensor.shape[0]):
        single_followup_img_tensor = followup_img_tensor[batch_idx:batch_idx+1].to("cuda")  # shape [1, C, H, W]
        with GradCAMPlusPlus(model=model, target_layers=target_layers) as cam:
            single_followup_grayscale_cam = cam(
                input_tensor=single_followup_img_tensor,
                targets=targets,
                aug_smooth=False,
                eigen_smooth=False,
            )
        followup_grayscale_cam.append(single_followup_grayscale_cam[0])  # shape [H, W]
    base_grayscale_cam = np.stack(base_grayscale_cam, axis=0)  # shape [B, H, W]
    followup_grayscale_cam = np.stack(followup_grayscale_cam, axis=0)  # shape [B, H, W]

    base_idxes = input_batch['base']['batch_idxes'].cpu().numpy()
    followup_idxes = input_batch['followup']['batch_idxes'].cpu().numpy()
    patient_ids = input_batch["patient_ids"]
    # print(f"Patient IDs: {patient_ids}, morph_score_base: {morph_score_base}, morph_score_followup: {morph_score_followup}, base_predictions: {base_predictions}, followup_predictions: {followup_predicitons}")
    for i, patient_id in enumerate(patient_ids):
        base_patient_img_ids = np.where(base_idxes == i)[0]
        followup_patient_img_ids = np.where(followup_idxes == i)[0]
        morph_score_base_patient = morph_score_base[i].cpu().numpy()
        morph_score_followup_patient = morph_score_followup[i].cpu().numpy()

        base_predictions_patient = base_predictions[base_patient_img_ids].cpu().numpy()
        followup_predictions_patient = followup_predicitons[followup_patient_img_ids].detach().cpu().numpy()
        # check if img_ids are empty and skip if so
        if len(base_patient_img_ids) == 0 or len(followup_patient_img_ids) == 0:
            print(f"Skipping patient {patient_id} for target {target_id} because of empty img ids.")
            continue
        plt_save_cam_on_image(base_grayscale_cam[base_patient_img_ids], base_img_tensor.cpu().numpy()[base_patient_img_ids], patient_id, target_id, "base", 
                              method="GradCAMpp", gt = morph_score_base_patient, pred=base_predictions_patient)
        plt_save_cam_on_image(followup_grayscale_cam[followup_patient_img_ids], followup_img_tensor.cpu().numpy()[followup_patient_img_ids], patient_id, target_id, "followup", 
                              method="GradCAMpp", gt = morph_score_followup_patient, pred=followup_predictions_patient)


def perform_integrated_gradients(model, test_loader, target_id):
    model.eval()

    ig = IntegratedGradients(model)

    input_batch = next(iter(test_loader))
    base_img_tensor = input_batch['base']['img']  # shape [B, C, H, W]
    followup_img_tensor = input_batch['followup']['img']  # shape [B, C, H, W]

    morph_score_base = input_batch['targets']['morph_score_base']  # shape [B]
    morph_score_followup = input_batch['targets']['morph_score_followup']  # shape [B]

    with torch.no_grad():
        base_predictions = torch.softmax(model(base_img_tensor.to("cuda")), dim=1)
        followup_predicitons = torch.softmax(model(followup_img_tensor.to("cuda")), dim=1)

    base_attr = []
    followup_attr = []

    for batch_idx in range(base_img_tensor.shape[0]):
        single_base_img_tensor = base_img_tensor[batch_idx:batch_idx+1].to("cuda")
        single_base_baseline = torch.zeros_like(single_base_img_tensor)

        single_base_attr = ig.attribute(
            single_base_img_tensor,
            baselines=single_base_baseline,
            target=target_id,
            n_steps=40
        )

        single_base_attr = single_base_attr.abs().sum(dim=1).detach().cpu().numpy()[0]  # [H, W]
        single_base_attr = single_base_attr - single_base_attr.min()
        single_base_attr = single_base_attr / (single_base_attr.max() + 1e-8)
        base_attr.append(single_base_attr)

    for batch_idx in range(followup_img_tensor.shape[0]):
        single_followup_img_tensor = followup_img_tensor[batch_idx:batch_idx+1].to("cuda")
        single_followup_baseline = torch.zeros_like(single_followup_img_tensor)

        single_followup_attr = ig.attribute(
            single_followup_img_tensor,
            baselines=single_followup_baseline,
            target=target_id,
            n_steps=40
        )

        single_followup_attr = single_followup_attr.abs().sum(dim=1).detach().cpu().numpy()[0]  # [H, W]
        single_followup_attr = single_followup_attr - single_followup_attr.min()
        single_followup_attr = single_followup_attr / (single_followup_attr.max() + 1e-8)
        followup_attr.append(single_followup_attr)

    base_attr = np.stack(base_attr, axis=0)  # [B, H, W]
    followup_attr = np.stack(followup_attr, axis=0)  # [B, H, W]

    base_idxes = input_batch['base']['batch_idxes'].cpu().numpy()
    followup_idxes = input_batch['followup']['batch_idxes'].cpu().numpy()
    patient_ids = input_batch["patient_ids"]

    for i, patient_id in enumerate(patient_ids):
        base_patient_img_ids = np.where(base_idxes == i)[0]
        followup_patient_img_ids = np.where(followup_idxes == i)[0]

        morph_score_base_patient = morph_score_base[i].cpu().numpy()
        morph_score_followup_patient = morph_score_followup[i].cpu().numpy()

        base_predictions_patient = base_predictions[base_patient_img_ids].cpu().numpy()
        followup_predictions_patient = followup_predicitons[followup_patient_img_ids].cpu().numpy()
        # check if img_ids are empty and skip if so
        if len(base_patient_img_ids) == 0 or len(followup_patient_img_ids) == 0:
            print(f"Skipping patient {patient_id} for target {target_id} because of empty img ids.")
            continue
        plt_save_cam_on_image(
            base_attr[base_patient_img_ids],
            base_img_tensor.cpu().numpy()[base_patient_img_ids],
            patient_id,
            target_id,
            "base",
            gt=morph_score_base_patient,
            pred=base_predictions_patient,
            method="integrated_gradients",
            img_weight=0.5
        )

        plt_save_cam_on_image(
            followup_attr[followup_patient_img_ids],
            followup_img_tensor.cpu().numpy()[followup_patient_img_ids],
            patient_id,
            target_id,
            "followup",
            gt=morph_score_followup_patient,
            pred=followup_predictions_patient,
            method="integrated_gradients",
            img_weight=0.5
        )

def main():
    model, test_loader = get_model_and_dataloaders(fold_index=2)
    
    # perform_gradcam(model, test_loader, target_id=0)
    # perform_gradcam(model, test_loader, target_id=1)
    # perform_gradcam(model, test_loader, target_id=2)

    perform_integrated_gradients(model, test_loader, target_id=0)
    perform_integrated_gradients(model, test_loader, target_id=1)
    perform_integrated_gradients(model, test_loader, target_id=2)

if __name__ == "__main__":
    main()