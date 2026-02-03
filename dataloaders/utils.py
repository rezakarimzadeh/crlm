import json
import yaml


def save_to_json(data, filename):
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)


def read_json(filename):
    with open(filename, 'r') as f:
        return json.load(f)
    

def read_yaml(yaml_path):
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    return data


def print_label_distribution(data, label_key):
    label_counts = {}
    for item in data:
        val = list(item.values())[0]
        label = val[label_key]
        if label in label_counts:
            label_counts[label] += 1
        else:
            label_counts[label] = 1
    print(f"Label distribution for key '{label_key}':")
    for label, count in label_counts.items():
        print(f"  Label {label}: {count} samples")
        print(f"  Label {label}: {count / len(data) * 100:.2f}%")