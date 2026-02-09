
import json
import yaml


def read_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def save_to_json(data, file_path):
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)

def read_3d_volume(img_path):
    import SimpleITK as sitk
    img = sitk.ReadImage(img_path)
    return img

def read_yaml(file_path):
    with open(file_path, 'r') as f:
        data = yaml.safe_load(f)
    return data