import os
from natsort import natsorted 
import torch

IMAGE_EXT = [".JPG", ".jpg", ".jpeg", ".webp", ".bmp", ".png"]

def find_classes(directory):
    
    # Get class names
    classes = natsorted(entry.name for entry in os.scandir(directory) if os.path.isdir(entry))

    # Raise an error if classes not found
    if not classes:
        raise FileNotFoundError(f"Couldn't find any classes in {directory}")
    
    # Dictionary of index labels
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}

    return (classes, class_to_idx)

def get_image_list(path):
    image_names = []
    for maindir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = os.path.join(maindir, filename)
            ext = os.path.splitext(apath)[1]
            if ext in IMAGE_EXT:
                image_names.append(apath)
    return image_names

def get_default_device():
    """ Pick GPU if available, else CPU """
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")
    
def to_device(data, device):
    """ Move tensors to chosen device """
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)



