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

    # Initialize CycleGAN Generators and Discriminators
    G_night_to_day = UNetGenerator().to(DEVICE)
    G_day_to_night = UNetGenerator().to(DEVICE)
    D_night = PatchDiscriminator().to(DEVICE)
    D_day = PatchDiscriminator().to(DEVICE)

    # Perceptual Loss Model
    feature_extractor = MegaDetectorFeatureExtractor(device=DEVICE)

    # Initialize loss functions MSE for adversarial, L1 for cycle and perceptual
    adv_criterion = nn.MSELoss()
    cycle_criterion = nn.L1Loss()
    perceptual_criterion = nn.L1Loss()

    # Adam optimizers with learning rate from config
    lr = config["lr"]
    opt_G = optim.Adam(
        list(G_night_to_day.parameters()) + list(G_day_to_night.parameters()),
        lr=lr, betas=(0.5, 0.999)
    )
    opt_D = optim.Adam(
        list(D_night.parameters()) + list(D_day.parameters()),
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

    # Create output directories
    os.makedirs("saved_models_perceptual", exist_ok=True)
    os.makedirs("saved_images_perceptual", exist_ok=True)

    n_epochs = config["n_epochs"]
    lambda_cycle = config["lambda_cycle"]
    lambda_perceptual = config["lambda_perceptual"]
    model_save_path = config["model_save_path"]
    image_save_path = config["image_save_path"]
    for epoch in range(n_epochs):
        loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{n_epochs}]")
        for real_night, real_day, _ in loop:
            real_night, real_day = real_night.to(DEVICE), real_day.to(DEVICE)

            opt_G.zero_grad()

            # Standard CycleGAN forward pass
            fake_day = G_night_to_day(real_night)
            fake_night = G_day_to_night(real_day)

            # Compute adversarial losses
            loss_G_adv_day = adv_criterion(D_day(fake_day), torch.ones_like(D_day(fake_day)))
            loss_G_adv_night = adv_criterion(D_night(real_night), torch.ones_like(D_night(real_night)))
            loss_G_adv = (loss_G_adv_night + loss_G_adv_day)

            # Compute Cycle-consistency losses
            recov_night = G_day_to_night(fake_day)
            recov_day = G_night_to_day(fake_night)
            loss_cycle = cycle_criterion(recov_night, real_night) + cycle_criterion(recov_day, real_day)

            # Compute perceptual loss between real_day and fake_day
            features_real_night = feature_extractor(real_night)
            features_fake_night = feature_extractor(fake_night)
            loss_perceptual = perceptual_criterion(features_fake_night, features_real_night)

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
            loss_D_night_real = adv_criterion(D_night(real_night), torch.ones_like(D_night(real_night)))
            loss_D_night_fake = adv_criterion(D_night(fake_night.detach()), torch.zeros_like(D_night(fake_night)))
            loss_D_night = (loss_D_night_real + loss_D_night_fake) * 0.5

            # D_Y
            loss_D_day_real = adv_criterion(D_day(real_day), torch.ones_like(D_day(real_day)))
            loss_D_day_fake = adv_criterion(D_day(fake_day.detach()), torch.zeros_like(D_day(fake_day)))
            loss_D_day = (loss_D_day_real + loss_D_day_fake) * 0.5

            # Total Discriminator Loss
            loss_D = loss_D_night + loss_D_day
            loss_D.backward()
            opt_D.step()

            loop.set_postfix({
                "loss_G": f"{loss_G.item():.3f}",
                "loss_D": f"{loss_D.item():.3f}",
                "loss_percept": f"{loss_perceptual.item():.3f}"
            })

        # --- Save example outputs at end of epoch ---
        # Use test_loader to get a consistent image for comparison across epochs
        val_real_night, _, _ = next(iter(test_loader))
        val_real_night = val_real_night.to(DEVICE)
        
        G_night_to_day.eval()
        G_day_to_night.eval()
        with torch.no_grad():
            val_fake_day = G_night_to_day(val_real_night)
            val_recov_night = G_day_to_night(val_fake_day)
        G_night_to_day.train()
        G_day_to_night.train()

        save_image(val_real_night * 0.5 + 0.5, os.path.join(image_save_path ,f"real_night_epoch{epoch}.png"))
        save_image(val_fake_day * 0.5 + 0.5, os.path.join(image_save_path ,f"fake_day_epoch{epoch}.png"))
        save_image(val_recov_night * 0.5 + 0.5, os.path.join(image_save_path ,f"val_recov_night_epoch{epoch}.png"))

        # --- Save model checkpoints ---
        if (epoch + 1) % 5 == 0:
            torch.save(G_night_to_day.state_dict(), os.path.join(model_save_path ,f"G_night_to_day_epoch{epoch}.pth"))
            torch.save(G_day_to_night.state_dict(), os.path.join(model_save_path ,f"G_day_to_night_epoch{epoch}.pth"))
            torch.save(D_night.state_dict(), os.path.join(model_save_path ,f"D_night_epoch{epoch}.pth"))
            torch.save(D_day.state_dict(), os.path.join(model_save_path ,f"D_day_epoch{epoch}.pth"))

    print("\nTraining complete.")

if __name__ == "__main__":
    main()
