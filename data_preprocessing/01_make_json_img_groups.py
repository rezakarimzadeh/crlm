import os
import json
from glob import glob
from tqdm import tqdm
import pathlib as Path
from utils import save_to_json, read_yaml


def mask_json_img_details(data_dir, output_json):
    img_details = []
    img_paths = sorted(glob(os.path.join(data_dir, '*.nii.gz')))
    img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_paths]
    patient_ids = set(sorted([img_id.split('_')[0] for img_id in img_ids]))
    print(f"Total unique patients: {len(patient_ids)}")
    print("number of images found:", len(img_ids))
    patients_with_multiple_followups = 0
    for patient_id in tqdm(patient_ids):
        img_group_ids =  [img_id for img_id in img_ids if img_id.startswith(patient_id)]
        follow_up_ids = []
        follow_ups_segs = []
        for img_id in img_group_ids:
            img_number = img_id.split('_')[1]
            if int(img_number) > 0:
                follow_up_ids.append(img_id)
                follow_ups_segs.append(patient_id + f'_{img_number}.nii.gz')
            else:
                base = img_id
                base_seg = patient_id + '_0.nii.gz'
        # skip the images without follow-ups
        if len(follow_up_ids) == 0:
            continue
        
        if len(follow_up_ids) > 1:
            patients_with_multiple_followups += 1
        detail = {
            'patient_id': patient_id,
            'base_img': base,
            'follow_ups_imgs': follow_up_ids,
            'all_imgs': [base] + follow_up_ids,
            'base_seg': base_seg,
            'follow_ups_segs': follow_ups_segs
        }

            
        img_details.append(detail)
    print(f"Total patients with follow-ups: {len(img_details)}")  
    print(f"Total patients with multiple follow-ups: {patients_with_multiple_followups}")              
    save_to_json(img_details, output_json)

if __name__ == "__main__":  
    data_config = read_yaml('../configs/data_config.yaml')
    data_directory = data_config['raw_images_dir']
    output_json_path = Path.Path(data_config['preprocessed_data_base_dir']) / "img_groups.json"
    os.makedirs(output_json_path.parent, exist_ok=True)
    mask_json_img_details(data_directory, output_json_path)

    # outcome:
    # Total unique patients: 445
    # number of images found: 1014
    # Total patients with follow-ups: 435
    # Total patients with multiple follow-ups: 117