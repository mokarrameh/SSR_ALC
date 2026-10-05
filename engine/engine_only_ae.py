"""
Contains functions for training and testing the ONLY_AE experiment.
"""
import torch
from .utils import save_model
from .utils import EarlyStoppingLoss


import os 

import copy

from tqdm.auto import tqdm
from typing import Tuple

import numpy as np

import csv

from .utils import display_random_images

# For colored terminal text
from colorama import Fore, Back, Style

# Constants and Configurations
c_ = Fore.GREEN # improvements
w_ = Fore.YELLOW # warning
e_ = Fore.RED # error
i_ = Fore.BLUE # info
g_ = Fore.CYAN # changes
b_ = Fore.MAGENTA # begining
sr_ = Style.RESET_ALL

def train_step(AE: torch.nn.Module, 
               dataloader_A: torch.utils.data.DataLoader, 
               loss_fn_A: torch.nn.Module, 
               optimizer_A: torch.optim.Optimizer,
               device: torch.device):
    """Trains a PyTorch model for a single epoch.

    Turns a target PyTorch model to training mode and then
    runs through all of the required training steps (forward
    pass, loss calculation, optimizer step).

    Args:
    model: A PyTorch model to be trained.
    dataloader: A DataLoader instance for the model to be trained on.
    loss_fn: A PyTorch loss function to minimize.
    optimizer: A PyTorch optimizer to help minimize the loss function.
    device: A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
    A tuple of training loss and training accuracy metrics.
    In the form (train_loss, train_accuracy). For example:

    (0.1112, 0.8743)
    """
    # Put model in train mode
    AE.train()

    # Setup train loss and train accuracy values
    AE_train_loss= 0

    # Loop through data loader data batches
    for data_A in dataloader_A:  # Loop based on the longest dataloader
        # Send data to target device
        data_A = data_A.to(device)

        # ------------------ AutoEncoder Training ------------------
        optimizer_A.zero_grad()
        
        _, decoded = AE(data_A)

        # Calculate and accumulate AE training loss
        AE_loss = loss_fn_A(decoded, data_A)
        AE_train_loss += AE_loss.item() 

        AE_loss.backward()

        torch.nn.utils.clip_grad_norm_(AE.parameters(), max_norm=1.0)  # Clip gradients for AE
        optimizer_A.step()

    # Adjust metrics to get average loss and accuracy per batch 
    AE_train_loss = AE_train_loss / len(dataloader_A)

    return AE_train_loss

def val_step(AE: torch.nn.Module, 
              dataloader_A: torch.utils.data.DataLoader, 
              loss_fn_A: torch.nn.Module,
              device: torch.device):
    """Tests a PyTorch model for a single epoch.

    Turns a target PyTorch model to "eval" mode and then performs
    a forward pass on a testing dataset.

    Args:
    model: A PyTorch model to be tested.
    dataloader: A DataLoader instance for the model to be tested on.
    loss_fn: A PyTorch loss function to calculate loss on the test data.
    device: A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
    A tuple of testing loss and testing accuracy metrics.
    In the form (test_loss, test_accuracy). For example:

    (0.0223, 0.8985)
    """
    # Put model in eval mode
    AE.eval() 

    # Setup test loss and test accuracy values
    AE_val_loss= 0

    # Turn on inference context manager
    with torch.inference_mode():
        for data_A in dataloader_A:  # Loop based on the longest dataloader
            # Send data to target device
            data_A = data_A.to(device)

            # 1. Forward pass
            _, decoded = AE(data_A)

            # 2. Calculate and accumulate loss
            AE_loss = loss_fn_A(decoded, data_A)
            AE_val_loss += AE_loss.item()

        # Adjust metrics to get average loss and accuracy per batch 
        AE_val_loss = AE_val_loss / len(dataloader_A)
    return AE_val_loss

def test_step(AE: torch.nn.Module, 
              dataloader_A: torch.utils.data.DataLoader, 
              loss_fn_A: torch.nn.Module,
              device: torch.device):
    """Tests a PyTorch model for a single epoch.

    Turns a target PyTorch model to "eval" mode and then performs
    a forward pass on a testing dataset.

    Args:
    model: A PyTorch model to be tested.
    dataloader: A DataLoader instance for the model to be tested on.
    loss_fn: A PyTorch loss function to calculate loss on the test data.
    device: A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
    A tuple of testing loss and testing accuracy metrics.
    In the form (test_loss, test_accuracy). For example:

    (0.0223, 0.8985)
    """
    # Put model in eval mode
    AE.eval() 

    # Setup test loss and test accuracy values
    AE_test_loss= 0

    # Turn on inference context manager
    with torch.inference_mode():
        for data_A in dataloader_A:  # Loop based on the longest dataloader
            # Send data to target device
            data_A = data_A.to(device)

            # 1. Forward pass
            _, decoded = AE(data_A)

            # 2. Calculate and accumulate loss
            AE_loss = loss_fn_A(decoded, data_A)
            AE_test_loss += AE_loss.item()

        # Adjust metrics to get average loss and accuracy per batch 
        AE_test_loss = AE_test_loss / len(dataloader_A)
    return AE_test_loss

def train(AE: torch.nn.Module, 
          train_dataloader_A: torch.utils.data.DataLoader,
          val_dataloader_A: torch.utils.data.DataLoader,
          test_dataloader_A: torch.utils.data.DataLoader, 
          optimizer_A: torch.optim.Optimizer,
          loss_fn_A: torch.nn.Module,
          epochs: int,
          device: torch.device,
          scheduler_A,
          training_progress_dir,
          model_weights_dir,
          fold_results,
          writer_A: torch.utils.tensorboard.writer.SummaryWriter=None,
          AE_lr=5e-4,
          patience=20):
    """Trains and tests a PyTorch model.

    Passes a target PyTorch models through train_step() and test_step()
    functions for a number of epochs, training and testing the model
    in the same epoch loop.

    Calculates, prints and stores evaluation metrics throughout.

    Args:
    model: A PyTorch model to be trained and tested.
    train_dataloader: A DataLoader instance for the model to be trained on.
    test_dataloader: A DataLoader instance for the model to be tested on.
    optimizer: A PyTorch optimizer to help minimize the loss function.
    loss_fn: A PyTorch loss function to calculate loss on both datasets.
    epochs: An integer indicating how many epochs to train for.
    device: A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
    A dictionary of training and testing loss as well as training and
    testing accuracy metrics. Each metric has a value in a list for 
    each epoch.
    In the form: {train_loss: [...],
              train_acc: [...],
              test_loss: [...],
              test_acc: [...]} 
    For example if training for epochs=2: 
             {train_loss: [2.0616, 1.0537],
              train_acc: [0.3945, 0.3945],
              test_loss: [1.2641, 1.5706],
              test_acc: [0.3400, 0.2973]} 
    """

    csv_file_path = model_weights_dir / "training_results.csv"
    # Initialize CSV file with header if it doesn't exist
    fieldnames = ["Epoch", "AE_train_loss", "AE_val_loss"]
    if not os.path.isfile(csv_file_path):
        with open(csv_file_path, "w", newline='') as csvfile:
            csv_writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            csv_writer.writeheader()
    
    best_loss  = np.inf
    best_epoch_ae = -1
    
    # Make sure model on target device
    AE.to(device)

    early_stopping_loss = EarlyStoppingLoss(patience=patience, min_delta=0.001)

    # Loop through training and testing steps for a number of epochs
    for epoch in tqdm(range(epochs)):

        AE_train_loss = train_step(AE=AE, 
                                    dataloader_A=train_dataloader_A,
                                    loss_fn_A=loss_fn_A,
                                    optimizer_A=optimizer_A,
                                    device=device)

        AE_val_loss = val_step(AE=AE, 
                                dataloader_A=val_dataloader_A,
                                loss_fn_A=loss_fn_A,
                                device=device)
        
        scheduler_A.step(AE_val_loss)

        # Print out what's happening
        print(
          f"{sr_}Epoch: {epoch+1}\n"
          f"\t|AE_train_loss: {AE_train_loss:.4f} | AE_val_loss: {AE_val_loss:.4f}"
        )

        # deep copy the model
        if AE_val_loss < best_loss:
            print(f"{c_}AE val Loss Improved ({best_loss:0.4f} ---> {AE_val_loss:0.4f})")
            best_loss = AE_val_loss
            best_epoch_ae = epoch+1
            
            # save validation output reconstruction for the current epoch
            display_random_images(data_loader=val_dataloader_A,
                                  AE=AE,
                                  file_recon=os.path.join(training_progress_dir, f"epoch{epoch + 1}_val_recon.png"),
                                  display_real=False,
                                  device=device)
            
            # Save the model
            AE_save_filepath = f"AE_{epochs}_epochs.pth"
            save_model(model_state_dict=AE.state_dict(),
                       target_dir=model_weights_dir,
                       model_name=AE_save_filepath)
            
            print(f"{sr_}")

        # Write the results to CSV file
        with open(csv_file_path, "a", newline='') as csvfile:
            csv_writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            row = {
                "Epoch": epoch + 1,
                "AE_train_loss": AE_train_loss,
                "AE_val_loss": AE_val_loss
            }
            csv_writer.writerow(row)

        # See if there's a writer, if so, log to it
        if writer_A:
            # Add results to SummaryWriter
            writer_A.add_scalars(main_tag="AE Loss", 
                                tag_scalar_dict={"AE_train_loss": AE_train_loss,
                                                "AE_val_loss": AE_val_loss},
                               global_step=epoch)

            # Close the writer
            writer_A.close()
        else:
            pass

        if scheduler_A.get_last_lr()[-1] != AE_lr:
            print(f"{g_}AE learning rate decreased: {AE_lr} -> {scheduler_A.get_last_lr()[-1]}")
            AE_lr = scheduler_A.get_last_lr()[-1]
            print(f"{sr_}")

        # Check early stopping conditions
        early_stopping_loss(AE_val_loss)

        if early_stopping_loss.early_stop:
            print(f"{w_}Early stopping at epoch {epoch}")
            break
    
    AE_test_loss = test_step(AE=AE, 
                                dataloader_A=test_dataloader_A,
                                loss_fn_A=loss_fn_A,
                                device=device)
    
    display_random_images(data_loader=test_dataloader_A,
                                  AE=AE,
                                  file_recon=os.path.join(training_progress_dir, f"epoch{epoch + 1}_test_recon.png"),
                                  display_real=False,
                                  device=device)
    
    fold_results["AE_test_loss"].append(AE_test_loss)

    print(f"{c_}[INFO] Final Results:")
    print(
          f"Best AE Epoch: {best_epoch_ae}\n"
          f"\t|Best AE loss: {best_loss:.4f}"
        )
    
    print(f"{c_}[INFO] Final Test Results:")
    print(
          f"\t|Test AE Loss: {AE_test_loss:.4f}"
        )

    print(f"{sr_}")
