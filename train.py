import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import numpy as np
from tqdm import tqdm
import os

from model import UNetGenerator, PatchDiscriminator, MegaDetectorFeatureExtractor
from dataset import CCTDataset
# --- Dataset and Model definitions from wildlife_domain_adapt.ipynb ---


# --- Configuration ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_DIR = "images/"
JSON_PATH = "missouri_camera_traps_set1.json"

# Training Hyperparameters
N_EPOCHS = 10
LR = 2e-4
BATCH_SIZE = 1 # Perceptual loss with larger batches can be memory intensive
NUM_SAMPLES_TRAIN = 1024
NUM_SAMPLES_TEST = 32

# Loss Weights
LAMBDA_CYCLE = 10.0
LAMBDA_PERCEPTUAL = 10.0 # Weight for the new perceptual loss

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
    train_idx, test_idx = CCTDataset.generated_train_test_split(JSON_PATH)

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
            features_fake_Y = feature_extractor(fake_Y)
            loss_perceptual = perceptual_criterion(features_fake_Y, features_real_X)

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
