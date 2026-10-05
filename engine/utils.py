"""
A series of helper functions used in the AutoEncoder implementation.
"""
import os
import random
import shutil

import torch
import torch.nn as nn
import torchvision

import matplotlib
matplotlib.use("agg")
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnnotationBbox, OffsetImage

from pathlib import Path

import numpy as np


from torch.utils.tensorboard import SummaryWriter

def extract_random_images(data_loader, 
                          num_images):
    # initialize empty lists to store all images and labels
    all_images = []
    # iterate through the data loader to get images and labels
    for images in data_loader:
        # append the current batch of images and labels to the respective lists
        all_images.append(images)
        # stop the iteration if the total number of images exceeds 1000
        if len(all_images) * data_loader.batch_size > 1000:
            break
    # concatenate all the images and labels tensors along the 0th dimension
    all_images = torch.cat(all_images, dim=0)
    # generate random indices for selecting a subset of images and labels
    random_indices = np.random.choice(len(all_images), num_images, replace=False)
    # use the random indices to extract the corresponding images and labels
    random_images = all_images[random_indices]
    # return the randomly selected images and labels to the calling function
    return random_images

def extract_random_images_labeled(data_loader, 
                          num_images):
    # initialize empty lists to store all images and labels
    all_images = []
    # iterate through the data loader to get images and labels
    for images, _ in data_loader:
        # append the current batch of images and labels to the respective lists
        all_images.append(images)
        # stop the iteration if the total number of images exceeds 1000
        if len(all_images) * data_loader.batch_size > 1000:
            break
    # concatenate all the images and labels tensors along the 0th dimension
    all_images = torch.cat(all_images, dim=0)
    # generate random indices for selecting a subset of images and labels
    random_indices = np.random.choice(len(all_images), num_images, replace=False)
    # use the random indices to extract the corresponding images and labels
    random_images = all_images[random_indices]
    # return the randomly selected images and labels to the calling function
    return random_images

def display_images(images, 
                   num_images_per_row, 
                   title, 
                   filename=None, 
                   show=True):
    # calculate the number of rows needed to display all the images
    num_rows = len(images) // num_images_per_row
    # create a grid of images using torchvision's make_grid function
    grid = torchvision.utils.make_grid(
        images.cpu(), nrow=num_images_per_row, padding=2, normalize=True
    )
    # convert the grid to a NumPy array and transpose it to
    # the correct dimensions
    grid_np = grid.numpy().transpose((1, 2, 0))
    # create a new figure with the appropriate size
    plt.figure(figsize=(num_images_per_row * 2, num_rows * 2))
    # show the grid of images
    plt.imshow(grid_np)
    # remove the axis ticks
    plt.axis("off")
    # set the title of the plot
    plt.title(title, fontsize=16)

        # add labels for each image in the grid
    for i in range(len(images)):
        # calculate the row and column of the current image in the grid
        row = i // num_images_per_row
        col = i % num_images_per_row
        # add the label name as text to the plot
        plt.text(
            col * (images.shape[3] + 2) + images.shape[3] // 2,
            (row + 1) * (images.shape[2] + 2) - 5,
            "",
            fontsize=12,
            ha="center",
            va="center",
            color="white",
            bbox=dict(facecolor="black", alpha=0.5, lw=0)
        )
    # if show is True, display the plot
    if show:
        plt.show()
    else:
        # otherwise, save the plot to a file and close the figure
        plt.savefig(filename, bbox_inches="tight")
        plt.close()

def display_random_images(data_loader,
                          device,
                          AE=None,
                          file_recon=None,
                          file_real=None,
                          title_recon=None,
                          title_real=None,
                          display_real=True,
                          num_images=32,
                          num_images_per_row=8):
    # extract a random subset of images and labels from the data loader
    random_images = extract_random_images(data_loader, num_images)

    # if an encoder and decoder are provided,
    # use them to generate reconstructions
    if AE is not None:
        # set the encoder and decoder to evaluation mode
        AE.eval()
        # move the random images to the appropriate device
        random_images = random_images.to(device)
        # generate embeddings and reconstruction for the random images using the model
        random_embeddings, random_reconstructions = AE(random_images)

        # display the reconstructed images
        display_images(random_reconstructions.cpu(),
                       num_images_per_row,
                       title_recon,
                       file_recon,
                       show=False)
        # if specified, also display the original images
        if display_real:
            display_images(random_images.cpu(),
                           num_images_per_row,
                           title_real,
                           file_real,
                           show=False)
    # if no encoder and decoder are provided, simply display the original images
    else:
        display_images(random_images, 
                       num_images_per_row, 
                       title="Real Images",
                       show=False)

def display_random_images_labeled(data_loader,
                          device,
                          AE=None,
                          file_recon=None,
                          file_real=None,
                          title_recon=None,
                          title_real=None,
                          display_real=True,
                          num_images=32,
                          num_images_per_row=8):
    # extract a random subset of images and labels from the data loader
    random_images = extract_random_images_labeled(data_loader, num_images)

    # if an encoder and decoder are provided,
    # use them to generate reconstructions
    if AE is not None:
        # set the encoder and decoder to evaluation mode
        AE.eval()
        # move the random images to the appropriate device
        random_images = random_images.to(device)
        # generate embeddings and reconstruction for the random images using the model
        random_embeddings, random_reconstructions = AE(random_images)

        # display the reconstructed images
        display_images(random_reconstructions.cpu(),
                       num_images_per_row,
                       title_recon,
                       file_recon,
                       show=False)
        # if specified, also display the original images
        if display_real:
            display_images(random_images.cpu(),
                           num_images_per_row,
                           title_real,
                           file_real,
                           show=False)
    # if no encoder and decoder are provided, simply display the original images
    else:
        display_images(random_images, 
                       num_images_per_row, 
                       title="Real Images",
                       show=False)

class ReplayBuffer():
    def __init__(self, max_size=50):
        assert (max_size > 0), 'Empty buffer or trying to create a black hole. Be careful.'
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):
        to_return = []
        for element in data:
            element = torch.unsqueeze(element.detach(), 0)  # Detach to avoid computation graph issues
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                if random.uniform(0, 1) > 0.5:
                    i = random.randint(0, self.max_size - 1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)
        return torch.cat(to_return, 0)  # Return a tensor directly without Variable
 
def save_model(model_state_dict,
               target_dir: str,
               model_name: str):
    """Saves a PyTorch model to a target directory with error handling."""
    
    # Create target directory
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True, exist_ok=True)

    # Validate model filename
    assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name should end with '.pt' or '.pth'"
    
    # Build full path
    model_save_path = target_dir_path / model_name

    # Save the model with error handling
    print(f"[INFO] Saving model to: {model_save_path}")
    try:
        torch.save(obj=model_state_dict, f=model_save_path)
        print(f"[SUCCESS] Model saved to {model_save_path}")
    except Exception as e:
        print(f"[ERROR] Failed to save model: {e}")
        total, used, free = shutil.disk_usage(str(target_dir_path))
        print(f"[INFO] Disk usage - Total: {total // (1024**2)} MB, Free: {free // (1024**2)} MB")

def create_output_dir (root_dir):
    # create output directory for autoencoder
    output_dir = root_dir
    if not root_dir.is_dir():
        print(f"[INFO] {root_dir} doesn't exist, creating it...")
        os.makedirs(output_dir, exist_ok=True)

    # create the training_progress directory inside the output directory
    training_progress_dir = output_dir/  "training_progress"
    if not training_progress_dir.is_dir():
        print(f"[INFO] {training_progress_dir} doesn't exist, creating it...")
        os.makedirs(training_progress_dir, exist_ok=True)
    # create the model_weights directory inside the output directory
    # for storing autoencoder weights
    model_weights_dir = output_dir / "model_weights"
    if not model_weights_dir.is_dir():
        print(f"[INFO] {model_weights_dir} doesn't exist, creating it...")
        os.makedirs(model_weights_dir, exist_ok=True)
    return training_progress_dir, model_weights_dir

def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

def create_writer(experiment_name: str, 
                  parent_dir: str, 
                  extra: str=None) -> torch.utils.tensorboard.writer.SummaryWriter:
    """Creates a torch.utils.tensorboard.writer.SummaryWriter() instance saving to a specific log_dir.

    log_dir is a combination of runs/timestamp/experiment_name/model_name/extra.

    Where timestamp is the current date in YYYY-MM-DD format.

    Args:
        experiment_name (str): Name of experiment.
        model_name (str): Name of model.
        extra (str, optional): Anything extra to add to the directory. Defaults to None.

    Returns:
        torch.utils.tensorboard.writer.SummaryWriter(): Instance of a writer saving to log_dir.

    Example usage:
        # Create a writer saving to "runs/2022-06-04/data_10_percent/effnetb2/5_epochs/"
        writer = create_writer(experiment_name="data_10_percent",
                               model_name="effnetb2",
                               extra="5_epochs")
        # The above is the same as:
        writer = SummaryWriter(log_dir="runs/2022-06-04/data_10_percent/effnetb2/5_epochs/")
    """
    from datetime import datetime
    import os

    # Get timestamp of current date (all experiments on certain day live in same folder)
    timestamp = datetime.now().strftime("%Y-%m-%d") # returns current date in YYYY-MM-DD format

    if extra:
        # Create log directory path
        log_dir = os.path.join(parent_dir,"runs", timestamp, experiment_name, extra)
    else:
        log_dir = os.path.join(parent_dir,"runs", timestamp, experiment_name)
        
    print(f"[INFO] Created SummaryWriter, saving to: {log_dir}...")
    return SummaryWriter(log_dir=log_dir)

class EarlyStoppingLoss:
    def __init__(self, patience=20, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

class EarlyStoppingF1:
    def __init__(self, patience=20, min_delta=0.01):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_f1 = None
        self.early_stop = False

    def __call__(self, val_f1):
        if self.best_f1 is None:
            self.best_f1 = val_f1
        elif val_f1 > self.best_f1 + self.min_delta:
            self.best_f1 = val_f1
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

class EarlyStoppingAcc:
    def __init__(self, patience=20, min_delta=0.01):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_acc = None
        self.early_stop = False

    def __call__(self, val_acc):
        if self.best_acc is None:
            self.best_acc = val_acc
        elif val_acc > self.best_acc + self.min_delta:
            self.best_acc = val_acc
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
