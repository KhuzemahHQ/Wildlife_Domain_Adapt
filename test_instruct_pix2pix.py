import torch
from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler
from PIL import Image
import numpy as np
import os
import json

# --- Configuration ---
# Model ID from Hugging Face Hub
model_id = "timbrooks/instruct-pix2pix"

# Data paths, mirroring the setup in wildlife_domain_adapt.ipynb
image_dir = "caltech_data/"
json_path = "caltech_data/missouri_camera_traps_set1.json"
output_folder = "results/cct_instructpix2pix/"

# The instruction prompt for the model. This guides the translation.
prompt = "turn this into a daytime photo"

# Inference parameters. You can tune these to adjust the output quality.
num_inference_steps = 20
image_guidance_scale = 1.5 # How much to follow the image structure.

# Limit the number of images to process for a quicker test run.
# Set to None to process all found images.
num_samples = 512
# --- End Configuration ---

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

def main():
    """
    Main function to run the image translation process using Instruct-Pix2Pix.
    """
    # Check for GPU availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Create output directory if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    print(f"Output images will be saved to: {output_folder}")

    # --- Load image paths from JSON, same as the notebook ---
    print(f"Loading image info from: {json_path}")
    with open(json_path, 'r') as f:
        ann = json.load(f)

    images_info = ann['images']
    image_paths = []
    print("Filtering for greyscale (nighttime) images...")

    for img_info in images_info:
        full_path = os.path.join(image_dir, img_info['file_name'])
        if not os.path.isfile(full_path):
            continue # Skip missing files

        try:
            with Image.open(full_path) as img:
                if check_greyscale(img):
                    image_paths.append(full_path)
        except Exception as e:
            print(f"Warning: Could not process {full_path}. Error: {e}")
            continue

        if num_samples is not None and len(image_paths) >= num_samples:
            print(f"Reached sample limit of {num_samples}.")
            break
    # --- End data loading logic ---

    # Load the pre-trained Instruct-Pix2Pix pipeline
    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        model_id, torch_dtype=torch.float16, safety_checker=None,
    )
    pipe.to(device)
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

    if not image_paths:
        print(f"Error: No greyscale images found based on '{json_path}'. Please check your data and paths.")
        return

    print(f"Found {len(image_paths)} images to process.")

    # Process each image
    for i, image_path in enumerate(image_paths):
        print(f"Processing image {i+1}/{len(image_paths)}: {image_path}")
        
        # Open the image and convert to RGB (Instruct-Pix2Pix expects RGB input)
        input_image = Image.open(image_path).convert("RGB")

        # Run inference
        # The generator helps ensure results are reproducible if you need them to be.
        generator = torch.Generator(device).manual_seed(42)
        edited_image = pipe(prompt, image=input_image, num_inference_steps=num_inference_steps, image_guidance_scale=image_guidance_scale, generator=generator).images[0]

        # Save the output image
        base_filename = os.path.basename(image_path)
        output_path = os.path.join(output_folder, base_filename)
        edited_image.save(output_path)

    print("\nProcessing complete.")

if __name__ == "__main__":
    main()