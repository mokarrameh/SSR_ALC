import os
import random
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torchvision.transforms as transforms
import numpy as np
import json

def show_class_samples(
    dataset_root,
    target_class,
    num_images=12,
    image_size=(128, 128),
    rows=3,
    cols=4,
    output_path=None
):
    """
    Displays and optionally saves a grid of sample images from a specific class.

    Parameters:
        dataset_root (str): Root path to the dataset directory.
        target_class (str): Name of the class to sample from.
        num_images (int): Number of images to display.
        image_size (tuple): Desired size (width, height) for each image.
        rows (int): Number of rows in the image grid.
        cols (int): Number of columns in the image grid.
        output_path (str, optional): If provided, saves the figure to this path.
    """
    ileum_path = os.path.join(dataset_root, target_class)
    image_files = [f for f in os.listdir(ileum_path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    
    if not image_files:
        raise ValueError(f"No image files found in {ileum_path}")
    
    selected_files = random.sample(image_files, min(num_images, len(image_files)))

    images = []
    for file in selected_files:
        img_path = os.path.join(ileum_path, file)
        img = Image.open(img_path).resize(image_size)
        images.append(img)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))

    for idx, ax in enumerate(axes.flat):
        ax.axis('off')
        if idx < len(images):
            ax.imshow(images[idx])

    plt.tight_layout(pad=0.2)

    if output_path:
        plt.savefig(output_path, format='svg', bbox_inches='tight', pad_inches=0.05)
    
    plt.show()

def visualize_real_and_reconstructed_separately(
    model,
    dataset_root,
    transform,
    selected_files,
    image_size=(256, 256),
    rows=4,
    cols=4,
    output_real_path=None,
    output_recon_path=None,
    device="cpu"
):
    """
    Displays and optionally saves two separate figures:
    1. Grid of original images
    2. Grid of corresponding reconstructed images (from a trained autoencoder)

    Parameters:
        model (torch.nn.Module): Trained autoencoder model.
        dataset_root (str): Root dataset directory.
        target_class (str): Class name to visualize.
        num_images (int): Number of images to show.
        image_size (tuple): Resize dimensions (width, height).
        rows (int): Number of rows for the grid.
        cols (int): Number of columns for the grid.
        output_real_path (str): Optional path to save real image grid.
        output_recon_path (str): Optional path to save reconstructed image grid.
        device (str): 'cuda' or 'cpu'.
    """
    model.eval()
    model.to(device)

    original_imgs = []
    input_tensors = []

    # Open and read the list of file paths
    with open(selected_files, "r") as f:
        selected = json.load(f)

    for file_path in selected:
        img_path = os.path.join(dataset_root, file_path)
        img = Image.open(img_path).convert("RGB")
        original_imgs.append(img.resize(image_size))
        input_tensors.append(transform(img))

    inputs = torch.stack(input_tensors).to(device)

    with torch.no_grad():
        model_output = model(inputs)
        if isinstance(model_output, tuple):
            outputs = model_output[1]  # use reconstructed part
        else:
            outputs = model_output

    # ---- Plot real images ----
    fig_real, axes_real = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    for i, ax in enumerate(axes_real.flat):
        ax.imshow(original_imgs[i])
        ax.axis('off')
    
    plt.tight_layout(pad=0.2)
    if output_real_path:
        plt.savefig(output_real_path, format='svg', bbox_inches='tight', pad_inches=0.05)
    plt.show()

    # ---- Plot reconstructed images ----
    fig_recon, axes_recon = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    for i, ax in enumerate(axes_recon.flat):
        recon = outputs[i].detach().cpu().clamp(0, 1)
        if recon.dim() == 3:
            recon_img = recon.permute(1, 2, 0).numpy()
        else:
            raise ValueError(f"Unexpected reconstructed shape: {recon.shape}")
        ax.imshow(recon_img)
        ax.axis('off')
    
    plt.tight_layout(pad=0.2)
    if output_recon_path:
        plt.savefig(output_recon_path, format='svg', bbox_inches='tight', pad_inches=0.05)
    plt.show()

def decode_and_visualize_latents(decoder, latents, rows=4, cols=4, output_recon_path=None, device='cpu'):
    decoder.eval()
    with torch.no_grad():
        if isinstance(latents, np.ndarray):
            latents = torch.tensor(latents).float()
        latents = latents.to(device)
        reconstructions = decoder(latents)
        reconstructions = reconstructions.cpu().clamp(0, 1)

    num_images = latents.shape[0]
    fig, axes = plt.subplots(rows, cols, figsize=(cols*2, rows*2))
    for i, ax in enumerate(axes.flat):
        if i >= num_images:
            break
        img = reconstructions[i].permute(1, 2, 0).numpy()
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout(pad=0.2)
    if output_recon_path:
        plt.savefig(output_recon_path, format='svg', bbox_inches='tight', pad_inches=0.05)
    plt.show()

# Choose and save the fixed list of image paths from a list of target classes
def save_fixed_image_paths(dataset_root, target_classes, num_images, output_json):
    all_image_paths = []

    for target_class in target_classes:
        class_path = os.path.join(dataset_root, target_class)
        if not os.path.isdir(class_path):
            continue  # skip if the class directory doesn't exist

        image_files = [
            f for f in os.listdir(class_path)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ]
        class_image_paths = [os.path.join(target_class, f) for f in image_files]
        all_image_paths.extend(class_image_paths)

    # Randomly sample the desired number of images from the combined list
    selected_paths = random.sample(all_image_paths, min(num_images, len(all_image_paths)))

    # Save to JSON
    with open(output_json, 'w') as f:
        json.dump(selected_paths, f)
