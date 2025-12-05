import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import numpy as np
from tqdm import tqdm
import os
import yaml

from model import UNetGenerator, PatchDiscriminator, MegaDetectorFeatureExtractor
from dataset import CCTDataset
# --- Dataset and Model definitions from wildlife_domain_adapt.ipynb ---


# --- Configuration ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def read_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        cf = yaml.safe_load(f)
    return cf

def main():
    config = read_config("config.yaml")
    image_dir = config["image_dir"]
    json_path = config["json_path"]
    config = config["train"]

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
    lr = config["lr"]
    opt_G = optim.Adam(
        list(G_XtoY.parameters()) + list(G_YtoX.parameters()),
        lr=lr, betas=(0.5, 0.999)
    )
    opt_D = optim.Adam(
        list(D_X.parameters()) + list(D_Y.parameters()),
        lr=lr, betas=(0.5, 0.999)
    )

    # --- DataLoaders ---
    # Load train/test indices from JSON
    train_idx, test_idx = CCTDataset.generated_train_test_split(json_path)

    # Create datasets
    train_dataset = CCTDataset(
        image_dir=image_dir,
        json_path=json_path,
        indices=train_idx,
        num_samples=config["n_samples_train"],
        image_size=512
    )
    test_dataset = CCTDataset(
        image_dir=image_dir,
        json_path=json_path,
        indices=test_idx,
        num_samples=config["n_samples_train"],
        image_size=512,
        mode="test"
    )
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # --- Create output directories ---
    os.makedirs("saved_models_perceptual", exist_ok=True)
    os.makedirs("saved_images_perceptual", exist_ok=True)

    n_epochs = config["n_epochs"]
    lambda_cycle = config["lambda_cycle"]
    lambda_perceptual = config["lambda_perceptual"]
    model_save_path = config["model_save_path"]
    image_save_path = config["image_save_path"]
    for epoch in range(n_epochs):
        loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{n_epochs}]")
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
                + lambda_cycle * loss_cycle
                + lambda_perceptual * loss_perceptual
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

        save_image(val_real_X * 0.5 + 0.5, os.path.join(image_save_path ,f"real_X_epoch{epoch}.png"))
        save_image(val_fake_Y * 0.5 + 0.5, os.path.join(image_save_path ,f"fake_Y_epoch{epoch}.png"))
        save_image(val_recov_X * 0.5 + 0.5, os.path.join(image_save_path ,f"recov_X_epoch{epoch}.png"))

        # --- Save model checkpoints ---
        if (epoch + 1) % 5 == 0:
            torch.save(G_XtoY.state_dict(), os.path.join(model_save_path ,f"G_XtoY_epoch{epoch}.pth"))
            torch.save(G_YtoX.state_dict(), os.path.join(model_save_path ,f"G_YtoX_epoch{epoch}.pth"))
            torch.save(D_X.state_dict(), os.path.join(model_save_path ,f"D_X_epoch{epoch}.pth"))
            torch.save(D_Y.state_dict(), os.path.join(model_save_path ,f"D_Y_epoch{epoch}.pth"))

    print("\nTraining complete.")

if __name__ == "__main__":
    main()
