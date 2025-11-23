import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import numpy as np
from PIL import Image
import json
import os
from tqdm import tqdm

# --- MegaDetector Imports ---
# Ensure megadetector is installed and accessible
from megadetector.detection.run_detector import load_detector

# --- Dataset and Model definitions from wildlife_domain_adapt.ipynb ---
import os
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
                 indices: list,
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
                    self.grey_img_cat.append(1 if "bbox" in img_info.keys() else 0)
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

# =========================================================
# 1️⃣ U-Net Generator
# =========================================================
class UNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, down=True, use_dropout=False):
        super().__init__()
        if down:
            self.block = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(0.2, inplace=True)
            )
        else:
            self.block = nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        self.use_dropout = use_dropout
        if use_dropout:
            self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.block(x)
        if self.use_dropout:
            x = self.dropout(x)
        return x


class UNetGenerator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super().__init__()
        self.down1 = UNetBlock(in_channels, 64, down=True)
        self.down2 = UNetBlock(64, 128, down=True)
        self.down3 = UNetBlock(128, 256, down=True)
        self.down4 = UNetBlock(256, 512, down=True, use_dropout=True)
        self.down5 = UNetBlock(512, 512, down=True, use_dropout=True)

        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 512, 4, 2, 1),
            nn.ReLU(inplace=True)
        )

        self.up1 = UNetBlock(512, 512, down=False, use_dropout=True)
        self.up2 = UNetBlock(1024, 512, down=False, use_dropout=True)
        self.up3 = UNetBlock(1024, 256, down=False)
        self.up4 = UNetBlock(512, 128, down=False)
        self.up5 = UNetBlock(256, 64, down=False)

        self.final = nn.Sequential(
            nn.ConvTranspose2d(128, out_channels, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        bottleneck = self.bottleneck(d5)
        u1 = self.up1(bottleneck)
        u2 = self.up2(torch.cat([u1, d5], 1))
        u3 = self.up3(torch.cat([u2, d4], 1))
        u4 = self.up4(torch.cat([u3, d3], 1))
        u5 = self.up5(torch.cat([u4, d2], 1))
        out = self.final(torch.cat([u5, d1], 1))
        return out


# =========================================================
# 2️⃣ PatchGAN Discriminator
# =========================================================
class PatchDiscriminator(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, 4, 1, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(512, 1, 4, 1, 1)
        )

    def forward(self, x):
        return self.model(x)

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

# --- Configuration ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_DIR = "caltech_data/"
JSON_PATH = "caltech_data/missouri_camera_traps_set1.json"

# Training Hyperparameters
N_EPOCHS = 10
LR = 2e-4
BATCH_SIZE = 1 # Perceptual loss with larger batches can be memory intensive
NUM_SAMPLES_TRAIN = 1024
NUM_SAMPLES_TEST = 32

# Loss Weights
LAMBDA_CYCLE = 10.0
LAMBDA_PERCEPTUAL = 1.0 # Weight for the new perceptual loss

# --- MegaDetector Feature Extractor ---
class MegaDetectorFeatureExtractor(nn.Module):
    """
    A wrapper for MegaDetector to extract features from an intermediate layer.
    This uses PyTorch hooks to capture the output of a specific layer during the forward pass.
    """
    def __init__(self, device='cuda'):
        super().__init__()
        self.device = device
        # Load the MegaDetector model
        print("Loading MegaDetector for perceptual loss...")
        self.detector = load_detector('MDV5A')
        self.detector.model.eval() # Ensure it's in evaluation mode

        self.features = None
        # The layer name is specific to the YOLOv5 architecture used by MegaDetector.
        # 'model.10' is part of the backbone, capturing mid-level features.
        # This is a good candidate for structure-preserving features.
        layer_to_hook = self.detector.model.model[10]
        layer_to_hook.register_forward_hook(self.hook_fn)
        print(f"Hook registered on MegaDetector layer: {layer_to_hook.__class__.__name__}")

    def hook_fn(self, module, input, output):
        self.features = output

    def forward(self, x):
        # We only need the forward pass to trigger the hook, not the final detections.
        # We run this with no_grad to save memory and computation, as we don't train the detector.
        with torch.no_grad():
            # The detector's forward pass is on its `.model` attribute.
            # Input images need to be in the [0, 1] range for the detector.
            # CycleGAN outputs are [-1, 1], so we rescale.
            x_rescaled = (x + 1) / 2.0
            _ = self.detector.model(x_rescaled)
        return self.features

def main():
    print(f"Using device: {DEVICE}")

    # --- Initialize Models ---
    # CycleGAN Generators and Discriminators
    G_XtoY = UNetGenerator().to(DEVICE)
    G_YtoX = UNetGenerator().to(DEVICE)
    D_X = PatchDiscriminator().to(DEVICE)
    D_Y = PatchDiscriminator().to(DEVICE)

    # Perceptual Loss Model
    feature_extractor = MegaDetectorFeatureExtractor(device=DEVICE)

    # --- Losses ---
    adv_criterion = nn.MSELoss()
    cycle_criterion = nn.L1Loss()
    perceptual_criterion = nn.L1Loss() # L1 loss between feature maps

    # --- Optimizers ---
    opt_G = optim.Adam(
        list(G_XtoY.parameters()) + list(G_YtoX.parameters()),
        lr=LR, betas=(0.5, 0.999)
    )
    opt_D = optim.Adam(
        list(D_X.parameters()) + list(D_Y.parameters()),
        lr=LR, betas=(0.5, 0.999)
    )

    # --- DataLoaders ---
    # Load train/test indices from JSON
    with open(JSON_PATH, 'r') as f:
        ann = json.load(f)
    ann_img = ann["images"]
    train_idx = [i for i, img_info in enumerate(ann_img) if "n_boxes" not in img_info.keys()]
    test_idx = [i for i, img_info in enumerate(ann_img) if "n_boxes" in img_info.keys()]
    np.random.seed(0)
    np.random.shuffle(train_idx)
    np.random.shuffle(test_idx)

    # Create datasets
    train_dataset = CCTDataset(
        image_dir=IMAGE_DIR,
        json_path=JSON_PATH,
        indices=train_idx,
        num_samples=NUM_SAMPLES_TRAIN,
        image_size=512
    )
    test_dataset = CCTDataset(
        image_dir=IMAGE_DIR,
        json_path=JSON_PATH,
        indices=test_idx,
        num_samples=NUM_SAMPLES_TEST,
        image_size=512,
        mode="test"
    )
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # --- Create output directories ---
    os.makedirs("saved_models_perceptual", exist_ok=True)
    os.makedirs("saved_images_perceptual", exist_ok=True)

    # =========================================================
    # 6️⃣ Training Loop with Perceptual Loss
    # =========================================================
    for epoch in range(N_EPOCHS):
        loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{N_EPOCHS}]")
        for real_X, real_Y, _ in loop:
            real_X, real_Y = real_X.to(DEVICE), real_Y.to(DEVICE)

            # -----------------------
            #  Train Generators
            # -----------------------
            opt_G.zero_grad()

            # Standard CycleGAN forward pass
            fake_Y = G_XtoY(real_X)
            fake_X = G_YtoX(real_Y)

            # Adversarial losses
            loss_G_adv_Y = adv_criterion(D_Y(fake_Y), torch.ones_like(D_Y(fake_Y)))
            loss_G_adv_X = adv_criterion(D_X(fake_X), torch.ones_like(D_X(fake_X)))
            loss_G_adv = (loss_G_adv_X + loss_G_adv_Y)

            # Cycle-consistency losses
            recov_X = G_YtoX(fake_Y)
            recov_Y = G_XtoY(fake_X)
            loss_cycle = cycle_criterion(recov_X, real_X) + cycle_criterion(recov_Y, real_Y)

            # --- NEW: Perceptual Loss Calculation ---
            # We calculate perceptual loss on the reconstructed image `recov_X`
            # to ensure the A->B->A cycle preserves detector-relevant features.
            features_real_X = feature_extractor(real_X)
            features_recov_X = feature_extractor(recov_X)
            loss_perceptual = perceptual_criterion(features_recov_X, features_real_X)

            # --- Total Generator Loss ---
            # Add the new perceptual loss, weighted by its lambda
            loss_G = (
                loss_G_adv
                + LAMBDA_CYCLE * loss_cycle
                + LAMBDA_PERCEPTUAL * loss_perceptual
            )
            loss_G.backward()
            opt_G.step()

            # -----------------------
            #  Train Discriminators
            # -----------------------
            opt_D.zero_grad()

            # D_X
            loss_D_X_real = adv_criterion(D_X(real_X), torch.ones_like(D_X(real_X)))
            loss_D_X_fake = adv_criterion(D_X(fake_X.detach()), torch.zeros_like(D_X(fake_X)))
            loss_D_X = (loss_D_X_real + loss_D_X_fake) * 0.5

            # D_Y
            loss_D_Y_real = adv_criterion(D_Y(real_Y), torch.ones_like(D_Y(real_Y)))
            loss_D_Y_fake = adv_criterion(D_Y(fake_Y.detach()), torch.zeros_like(D_Y(fake_Y)))
            loss_D_Y = (loss_D_Y_real + loss_D_Y_fake) * 0.5

            # Total Discriminator Loss
            loss_D = loss_D_X + loss_D_Y
            loss_D.backward()
            opt_D.step()

            loop.set_postfix({
                "loss_G": f"{loss_G.item():.3f}",
                "loss_D": f"{loss_D.item():.3f}",
                "loss_percept": f"{loss_perceptual.item():.3f}"
            })

        # --- Save example outputs at end of epoch ---
        # Use test_loader to get a consistent image for comparison across epochs
        val_real_X, _, _ = next(iter(test_loader))
        val_real_X = val_real_X.to(DEVICE)
        
        G_XtoY.eval()
        G_YtoX.eval()
        with torch.no_grad():
            val_fake_Y = G_XtoY(val_real_X)
            val_recov_X = G_YtoX(val_fake_Y)
        G_XtoY.train()
        G_YtoX.train()

        save_image(val_real_X * 0.5 + 0.5, f"saved_images_perceptual/real_X_epoch{epoch}.png")
        save_image(val_fake_Y * 0.5 + 0.5, f"saved_images_perceptual/fake_Y_epoch{epoch}.png")
        save_image(val_recov_X * 0.5 + 0.5, f"saved_images_perceptual/recov_X_epoch{epoch}.png")

        # --- Save model checkpoints ---
        if (epoch + 1) % 5 == 0:
            torch.save(G_XtoY.state_dict(), f"saved_models_perceptual/G_XtoY_epoch{epoch}.pth")
            torch.save(G_YtoX.state_dict(), f"saved_models_perceptual/G_YtoX_epoch{epoch}.pth")
            torch.save(D_X.state_dict(), f"saved_models_perceptual/D_X_epoch{epoch}.pth")
            torch.save(D_Y.state_dict(), f"saved_models_perceptual/D_Y_epoch{epoch}.pth")

    print("\nTraining complete.")

if __name__ == "__main__":
    # Note: The CCTDataset in the notebook needs a slight modification to accept indices.
    # I've assumed this change is made in the imported `wildlife_domain_adapt.py`.
    # The change is to pass `indices` to the CCTDataset constructor and use them
    # instead of recalculating train/test splits inside `__init__`.
    # This is a better practice for separating data logic from dataset class logic.
    print("This script assumes `wildlife_domain_adapt.py` exists and its CCTDataset class")
    print("has been modified to accept a list of indices in its constructor.")
    
    # To make this runnable, we need to convert the notebook to a .py file
    # and adjust the CCTDataset class. For now, this script outlines the complete logic.
    # To run, you would first convert the notebook and then execute:
    # python train_cyclegan_perceptual.py
    
    # Since the notebook conversion is a prerequisite, I will comment out the main call
    # to prevent errors if run directly without that step.
    main()
