import simpleitk as sitk
import numpy as np
from Pathlib import Path
from utils import read_json


def resample_img_seg(img, new_spacing=[1.0, 1.0, 1.0], is_label=False):
    original_size = img.GetSize()
    original_spacing = img.GetSpacing()
    
    new_size = [
        int(np.round(original_size[i] * (original_spacing[i] / new_spacing[i])))
        for i in range(3)
    ]
    
    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(new_spacing)
    resample.SetSize(new_size)
    resample.SetOutputDirection(img.GetDirection())
    resample.SetOutputOrigin(img.GetOrigin())
    
    if is_label:
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resample.SetInterpolator(sitk.sitkBilinear)
    
    resampled_img = resample.Execute(img)

    return resampled_img

def rigid_registration(fixed_img, moving_img):
    initial_transform = sitk.CenteredTransformInitializer(
        fixed_img, 
        moving_img, 
        sitk.Euler3DTransform(), 
        sitk.CenteredTransformInitializerFilter.GEOMETRY
    )
    
    registration_method = sitk.ImageRegistrationMethod()
    registration_method.SetMetricAsMeanSquares()
    registration_method.SetOptimizerAsRegularStepGradientDescent(
        learningRate=1.0, 
        minStep=1e-6, 
        numberOfIterations=200
    )
    registration_method.SetInterpolator(sitk.sitkLinear)
    registration_method.SetInitialTransform(initial_transform, inPlace=False)
    
    final_transform = registration_method.Execute(fixed_img, moving_img)
    
    moving_resampled = sitk.Resample(
        moving_img, 
        fixed_img, 
        final_transform, 
        sitk.sitkLinear, 
        0.0, 
        moving_img.GetPixelID()
    )
    
    return moving_resampled, final_transform

def perform_transformation_to_mask(mask_img, transform, reference_img):
    resampled_mask = sitk.Resample(
        mask_img, 
        reference_img, 
        transform, 
        sitk.sitkNearestNeighbor, 
        0, 
        mask_img.GetPixelID()
    )
    return resampled_mask



def main(img_groups_dir, raw_images_dir, raw_masks_dir, output_dir):
    img_groups = read_json(img_groups_dir)
    output_dir = Path(output_dir)/"01_resampled_rigid_registered"
    output_dir.mkdir(parents=True, exist_ok=True)

    for group in img_groups:
        base_img_id = group['base_img']
        follow_ups = group['follow_ups']

        base_img_path = Path(raw_images_dir) / base_img_id
        base_mask_path = Path(raw_masks_dir) / base_img_id

        base_img = read_3d_volume(str(base_img_path))
        base_mask = read_3d_volume(str(base_mask_path))

        base_img_resampled = resample_img_seg(base_img, new_spacing=[1.0, 1.0, 1.0], is_label=False)
        base_mask_resampled = resample_img_seg(base_mask, new_spacing=[1.0, 1.0, 1.0], is_label=True)

        sitk.WriteImage(base_img_resampled, str(output_dir / f"{base_img_id}_resampled.nii.gz"))
        sitk.WriteImage(base_mask_resampled, str(output_dir / f"{base_img_id}_mask_resampled.nii.gz"))

        for follow_up_id in follow_ups:
            follow_up_img_path = Path(raw_images_dir) / f"{follow_up_id}.nii.gz"
            follow_up_mask_path = Path(raw_masks_dir) / f"{follow_up_id}_mask.nii.gz"

            follow_up_img = read_3d_volume(str(follow_up_img_path))
            follow_up_mask = read_3d_volume(str(follow_up_mask_path))

            follow_up_img_resampled = resample_img_seg(follow_up_img, new_spacing=[1.0, 1.0, 1.0], is_label=False)
            follow_up_mask_resampled = resample_img_seg(follow_up_mask, new_spacing=[1.0, 1.0, 1.0], is_label=True)

            registered_img, transform = rigid_registration(base_img_resampled, follow_up_img_resampled)
            registered