import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import numpy as np
from PIL import Image
import json
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import yaml

from megadetector.detection.run_detector import load_detector
from megadetector.visualization import visualization_utils as vis_utils

from model import UNetGenerator
from dataset import CCTDataset, check_greyscale

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def read_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        cf = yaml.safe_load(f)
    return cf

# MegaDetector output mapping
cat_map = {"1": "Animal", "2": "Person", "3": 'Vehicle'}

def draw_result(img, result, path, display=False):
    """
    Draws bounding boxes from MegaDetector results onto an image and saves it.
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(img)
    for det in result:
        x1, y1, w, h = np.array(det["bbox"]) * img.size[0]
        acc = det["conf"]
        cat = cat_map.get(det["category"], "Unknown")
        rect = patches.Rectangle((x1, y1), w, h, linewidth=2, edgecolor='red', facecolor='none')
        ax.text(x1 + 7, y1 - 10, f"{cat}: {acc*100:.1f}%", color="black", fontsize=10, backgroundcolor="red")
        ax.add_patch(rect)
    ax.axis("off")
    plt.savefig(path, bbox_inches="tight", pad_inches=0)
    if display:
        plt.show()
    else:
        plt.close(fig)

def main():
    """
    Main function to evaluate the CycleGAN model on the downstream detection task.
    """
    config = read_config("config.yaml")
    image_dir = config["image_dir"]
    json_path = config["json_path"]
    config = config["eval"]
    print(f"Using device: {DEVICE}")

    # Create output directory
    eval_path = config["eval_path"]
    os.makedirs(eval_path, exist_ok=True)
    print(f"Evaluation outputs will be saved to: {eval_path}")

    # Load MegaDetector
    print("Loading MegaDetector for evaluation...")
    detector = load_detector('MDV5A')

    # Load Generator Models
    model_path_G_night_to_day = config["model_path_G_night_to_day"]
    model_path_G_day_to_night = config["model_path_G_day_to_night"]
    if not os.path.exists(model_path_G_night_to_day):
        print(f"Error: Model file not found at {model_path_G_night_to_day}")
        print("Please ensure you have trained the model using train_cyclegan_perceptual.py first.")
        return
    
    if not os.path.exists(model_path_G_night_to_day):
        print(f"Error: Model file not found at {model_path_G_night_to_day}")
        print("Please ensure you have trained the model using train_cyclegan_perceptual.py first.")
        return

    print(f"Loading generator model from: {model_path_G_night_to_day} and {model_path_G_day_to_night}")
    G_night_to_day = UNetGenerator().to(DEVICE)
    G_day_to_night = UNetGenerator().to(DEVICE)
    G_night_to_day.load_state_dict(torch.load(model_path_G_night_to_day, map_location=DEVICE))
    G_day_to_night.load_state_dict(torch.load(model_path_G_day_to_night, map_location=DEVICE))
    G_night_to_day.eval()
    G_day_to_night.eval()

    # Create test dataset only
    _, test_idx = CCTDataset.generated_train_test_split(json_path)

    test_dataset = CCTDataset(
        image_dir=image_dir,
        json_path=json_path,
        indices=test_idx,
        num_samples=config["n_samples_test"],
        image_size=512,
        mode="test"
    )
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False)

    if len(test_dataset) == 0:
        print("Error: Test dataset is empty. Check paths and data curation.")
        return

    # Evaluation Loop
    print(f"\nStarting evaluation on {len(test_dataset)} test images...")
    # [True Neg, False Neg, False Pos, True Pos]
    stats_real_night = [0, 0, 0, 0]  # For original NIR images
    stats_fake_day = [0, 0, 0, 0]  # For generated daytime images
    stats_rec_night = [0, 0, 0, 0]   # For reconstructed NIR images

    loop = tqdm(enumerate(test_loader), total=len(test_loader), desc="Evaluating")
    detection_threshold = config["detection_threshold"]
    for i, (real_night, _, cat_night) in loop:
        real_night = real_night.to(DEVICE)
        
        # Generate fake images and reconstructed images 
        with torch.no_grad():
            fake_day = G_night_to_day(real_night)
            rec_night = G_day_to_night(fake_day)

        # Save images for MegaDetector to process
        real_night_path = os.path.join(eval_path, f"{i:03d}_real_night.png")
        fake_day_path = os.path.join(eval_path, f"{i:03d}_fake_day.png")
        rec_night_path = os.path.join(eval_path, f"{i:03d}_rec_night.png")
        save_image(real_night * 0.5 + 0.5, real_night_path)
        save_image(fake_day * 0.5 + 0.5, fake_day_path)
        save_image(rec_night * 0.5 + 0.5, rec_night_path)

        # Load images in a format suitable for MegaDetector
        img_night_pil = vis_utils.load_image(real_night_path)
        img_day_pil = vis_utils.load_image(fake_day_path)
        img_rec_pil = vis_utils.load_image(rec_night_path)

        # Run detection
        result_night = detector.generate_detections_one_image(img_night_pil, real_night_path)
        result_day = detector.generate_detections_one_image(img_day_pil, fake_day_path)
        result_rec = detector.generate_detections_one_image(img_rec_pil, rec_night_path)

        # Ground truth of animals in the images
        ground_truth_has_animal = cat_night.cpu().numpy()[0] > 0

        # Filter detections by confidence threshold
        detections_night = [d for d in result_night['detections'] if d['conf'] > detection_threshold]
        detections_day = [d for d in result_day['detections'] if d['conf'] > detection_threshold]
        detections_rec = [d for d in result_rec['detections'] if d['conf'] > detection_threshold]


        # Draw results on images
        draw_result(img_night_pil, detections_night, os.path.join(eval_path, f"{i:03d}_real_night_det.png"))
        draw_result(img_day_pil, detections_day, os.path.join(eval_path, f"{i:03d}_fake_day_det.png"))
        draw_result(img_rec_pil, detections_rec, os.path.join(eval_path, f"{i:03d}_rec_night_det.png"))

        # Update results which maps to the [TN, FN, FP, TP]
        stats_real_night[(len(detections_night) > 0) * 2 + ground_truth_has_animal] += 1
        stats_fake_day[(len(detections_day) > 0) * 2 + ground_truth_has_animal] += 1
        stats_rec_night[(len(detections_rec) > 0) * 2 + ground_truth_has_animal] += 1

    # Calculate metrics accuracy, precision, recall, F1-score
    print("\n--- Evaluation Results ---")

    def calculate_metrics(stats, name):
        tn, fn, fp, tp = stats
        total = sum(stats)
        if total == 0:
            print(f"\nNo data for {name} images.")
            return

        accuracy = (tp + tn) / total if total > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        print(f"\nMetrics for {name} images:")
        print(f"  - True Positives:  {tp}")
        print(f"  - False Positives: {fp}")
        print(f"  - True Negatives:  {tn}")
        print(f"  - False Negatives: {fn}")
        print(f"  --------------------")
        print(f"  - Accuracy:  {accuracy:.3f}")
        print(f"  - Precision: {precision:.3f}")
        print(f"  - Recall:    {recall:.3f}")
        print(f"  - F1 Score:  {f1_score:.3f}")

    calculate_metrics(stats_real_night, "Original NIR (real_night)")
    calculate_metrics(stats_fake_day, "Generated Daytime (fake_day)")
    calculate_metrics(stats_rec_night, "Reconstructed NIR (rec_night)")

    print("\nEvaluation complete.")

if __name__ == "__main__":
    main()
