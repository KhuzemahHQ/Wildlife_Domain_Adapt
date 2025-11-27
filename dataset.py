import os
import numpy as np
from PIL import Image
import json
from torch.utils.data import Dataset
import torchvision.transforms as T

class CCTDataset(Dataset):
    """
    Unpaired dataset loader for Caltech Camera Traps for CycleGAN-style training.
    Domain A / Domain B splitting is done by image path or metadata.
    """
    def __init__(self,
                 image_dir: str,
                 json_path: str,
                 indices: list[int],
                 num_samples: int = None,
                 transform: T.Compose = None,
                 image_size: int = 1024,
                 mode: str = 'train'):
        """
        Args:
          image_dir: root directory containing image files (subfolders or flat).
          json_path: full path to COCO-style JSON file with metadata.
          indices: List of integer indices to use for this dataset split.
          transform: torchvision transforms to apply.
          image_size: target size (will do Resize + CenterCrop or so).
          mode: 'train' or 'val' (if you want different behavior).
        """
        super().__init__()
        self.image_dir = image_dir
        self.transform = transform or T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.mode = mode

        # Load the annotation JSON
        with open(json_path, 'r') as f:
            ann = json.load(f)

        # Build image info list
        # Each item: {'file_name': ..., 'location': <int or str>, ...}
        images_info = ann['images']

        # Filter images by location into two domains
        self.grey_img = []
        self.len_grey = 0
        self.col_img = []
        self.len_col = 0
        self.grey_img_cat = []
        selected_indices = indices
        for idx in selected_indices:
            img_info = images_info[idx]
            # full path
            fn = img_info['file_name']
            full_path = os.path.join(image_dir, fn)
            if not os.path.isfile(full_path):
                # skip missing files
                continue
            if check_greyscale(Image.open(full_path)):
                if num_samples is None or self.len_grey < num_samples:
                    self.grey_img_cat.append(img_info["n_boxes"] if "n_boxes" in img_info.keys() else 0)
                    self.grey_img.append(full_path)
                    self.len_grey += 1
            else:
                if num_samples is None or self.len_col < num_samples:
                    self.col_img.append(full_path)
                    self.len_col += 1
            if num_samples is None:
                continue
            elif self.len_col >= num_samples and self.len_grey >= num_samples:
                break

        self.dataset_length = max(self.len_grey, self.len_col)
        print(f"Greyscale img count: {self.len_grey}, Colour img count: {self.len_col}, using dataset length {self.dataset_length}")
        
    def __len__(self):
        return self.dataset_length

    def __getitem__(self, idx):
        # For unpaired: sample A and B independently (or aligned by idx mod list length)
        pathA = self.grey_img[idx % self.len_grey]
        pathB = self.col_img[idx % self.len_col]
        catA = self.grey_img_cat[idx % self.len_grey]

        img_grey = Image.open(pathA).convert('RGB')
        img_col = Image.open(pathB).convert('RGB')

        img_grey = self.transform(img_grey)
        img_col = self.transform(img_col)

        return img_grey, img_col, catA
    
    @staticmethod
    def generated_train_test_split(json_path): 
        with open(json_path, 'r') as f:
            ann = json.load(f)
        ann_img = ann["images"]
        train_idx = [i for i, img_info in enumerate(ann_img) if "n_boxes" not in img_info.keys()]
        test_idx = [i for i, img_info in enumerate(ann_img) if "n_boxes" in img_info.keys()]
        np.random.seed(0)
        np.random.shuffle(train_idx)
        np.random.shuffle(test_idx)
        return train_idx, test_idx
    

def check_greyscale(img, threshold=1):
    """
    Helper function to check if a PIL image is greyscale.
    From wildlife_domain_adapt.ipynb.
    """
    arr = np.asarray(img, dtype=np.float32)
    # If the image has only one channel, it's greyscale
    if arr.ndim < 3:
        return True
    # If it has 3 channels, check if R, G, and B are very close
    diff_rg = np.abs(arr[...,0] - arr[...,1])
    diff_rb = np.abs(arr[...,0] - arr[...,2])
    diff_gb = np.abs(arr[...,1] - arr[...,2])
    mean_diff = (diff_rg.mean() + diff_rb.mean() + diff_gb.mean()) / 3.0
    return mean_diff < threshold


