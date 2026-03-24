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

    best_model_path = Path("Results") / "MorphScoreGooglenetLSTM" / f"fold_{fold_index}" / "checkpoints" / "last.ckpt"
    pl_model = MorphScoreGooglenetLSTM.load_from_checkpoint(best_model_path, config_dir=model_config_dir, weights_only=False)
    model = MorphScoreBaseBranch(pl_model)
    model.eval()

    train_loader, _, test_loader = get_cnn_dataloaders(data_config_dir, model_config_dir, fold_index)
    return model, train_loader

def plt_save_cam_on_image(cam_maps, rgb_imgs, patient_id, target_id, img_type, method="GradCAMpp"):
    plt.figure(figsize=(3,3*len(cam_maps)))
    plt.suptitle(f"Patient {patient_id}, target {target_id}", fontsize=16)
    plt.subplots(1, len(cam_maps), figsize=(15, 10))
    for i, (cam_map, rgb_img) in enumerate(zip(cam_maps, rgb_imgs)):
        # cam_map = cam_map.cpu().numpy()  # [H, W]
        # rgb_img = rgb_img.permute(1,2,0).cpu().numpy()  # [H, W, C]
        rgb_img =  np.transpose(rgb_img, (1,2,0))  # [H, W, C]
        rgb_img = (rgb_img - rgb_img.min()) / (rgb_img.max() - rgb_img.min())  # normalize to [0,1]
        visualization = show_cam_on_image(rgb_img, cam_map, use_rgb=True)
        plt.subplot(1, len(cam_maps), i+1)
        plt.imshow(visualization)
        plt.axis('off')
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


    with GradCAMPlusPlus(model=model, target_layers=target_layers) as cam:
        base_grayscale_cam = cam(
            input_tensor=base_img_tensor,
            targets=targets,
            aug_smooth=False,
            eigen_smooth=False,
        )

    with GradCAMPlusPlus(model=model, target_layers=target_layers) as cam:
        followup_grayscale_cam = cam(
            input_tensor=followup_img_tensor,
            targets=targets,
            aug_smooth=False,
            eigen_smooth=False,
        )

    base_idxes = input_batch['base']['batch_idxes'].cpu().numpy()
    followup_idxes = input_batch['followup']['batch_idxes'].cpu().numpy()
    patient_ids = input_batch["patient_ids"]
    for i, patient_id in enumerate(patient_ids):
        base_patient_img_ids = np.where(base_idxes == i)[0]
        followup_patient_img_ids = np.where(followup_idxes == i)[0]
        plt_save_cam_on_image(base_grayscale_cam[base_patient_img_ids], base_img_tensor.cpu().numpy()[base_patient_img_ids], patient_id, target_id, "base")
        plt_save_cam_on_image(followup_grayscale_cam[followup_patient_img_ids], followup_img_tensor.cpu().numpy()[followup_patient_img_ids], patient_id, target_id, "followup")

def perform_integrated_gradients(model, test_loader, target_id):
    model.eval()


    ig = IntegratedGradients(model)

    input_batch = next(iter(test_loader))
    base_img_tensor = input_batch['base']['img'].to("cuda")         # [B, C, H, W]
    followup_img_tensor = input_batch['followup']['img'].to("cuda") # [B, C, H, W]

    # baseline: zeros
    base_baseline = torch.zeros_like(base_img_tensor)
    followup_baseline = torch.zeros_like(followup_img_tensor)

    base_attr = ig.attribute(
        base_img_tensor,
        baselines=base_baseline,
        target=target_id,
        n_steps=40
    )

    followup_attr = ig.attribute(
        followup_img_tensor,
        baselines=followup_baseline,
        target=target_id,
        n_steps=40
    )

    # reduce channel dimension for visualization
    base_attr = base_attr.abs().sum(dim=1).detach().cpu().numpy()         # [B, H, W]
    followup_attr = followup_attr.abs().sum(dim=1).detach().cpu().numpy() # [B, H, W]

    base_idxes = input_batch['base']['batch_idxes'].cpu().numpy()
    followup_idxes = input_batch['followup']['batch_idxes'].cpu().numpy()
    patient_ids = input_batch["patient_ids"]

    for i, patient_id in enumerate(patient_ids):
        base_patient_img_ids = np.where(base_idxes == i)[0]
        followup_patient_img_ids = np.where(followup_idxes == i)[0]

        plt_save_cam_on_image(
            base_attr[base_patient_img_ids],
            input_batch['base']['img'].cpu().numpy()[base_patient_img_ids],
            patient_id,
            target_id,
            "base",
            "integrated_gradients"

        )

        plt_save_cam_on_image(
            followup_attr[followup_patient_img_ids],
            input_batch['followup']['img'].cpu().numpy()[followup_patient_img_ids],
            patient_id,
            target_id,
            "followup",
            "integrated_gradients"
        )

def main():
    model, test_loader = get_model_and_dataloaders(fold_index=3)
    
    perform_gradcam(model, test_loader, target_id=0)
    perform_gradcam(model, test_loader, target_id=1)
    perform_gradcam(model, test_loader, target_id=2)

    perform_integrated_gradients(model, test_loader, target_id=0)
    perform_integrated_gradients(model, test_loader, target_id=1)
    perform_integrated_gradients(model, test_loader, target_id=2)

if __name__ == "__main__":
    main()