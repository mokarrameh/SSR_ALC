"""
Contains functions for training and testing the experiment 5.
"""
import torch
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score, MulticlassAUROC, MulticlassPrecision, MulticlassRecall
from .utils import ReplayBuffer
from .utils import save_model
from .utils import EarlyStoppingF1, EarlyStoppingLoss


import os 

from tqdm.auto import tqdm
from typing import Tuple

import numpy as np

import csv

from .utils import display_random_images
import itertools

# For colored terminal text
from colorama import Fore, Style

# Constants and Configurations
c_ = Fore.GREEN # improvements
w_ = Fore.YELLOW # warning
e_ = Fore.RED # error
i_ = Fore.BLUE # info
g_ = Fore.CYAN # changes
b_ = Fore.MAGENTA # begining
sr_ = Style.RESET_ALL


def lambda_schedule(epoch, max_epochs, initial_lambda=1.0, final_lambda=10.0, growth_factor=1.0):
    # Ensure the input to torch.exp() is a tensor
    epoch_tensor = torch.tensor(epoch / max_epochs, dtype=torch.float32)
    exp_term = torch.exp(-growth_factor * epoch_tensor)
    return initial_lambda + (final_lambda - initial_lambda) * (1 - exp_term)

def train_step(AE: torch.nn.Module, 
               classifier: torch.nn.Module, 
               dataloader_A: torch.utils.data.DataLoader, 
               dataloader_B: torch.utils.data.DataLoader, 
               loss_fn_A: torch.nn.Module, 
               loss_fn_B: torch.nn.Module, 
               loss_fn_C: torch.nn.Module,
               optimizer_A: torch.optim.Optimizer,
               optimizer_B: torch.optim.Optimizer,
               device: torch.device,
               num_classes=6,
               alpha=1,
               beta=10,
               gamma=5):
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
    classifier.train()

    # Setup train loss and train accuracy values
    AE_train_loss= 0
    train_loss = 0

    # Instantiate Multiclass Accuracy instance
    acc = MulticlassAccuracy(num_classes=num_classes, average='macro').to(device)
    acc_mca = MulticlassAccuracy(num_classes=num_classes, average=None).to(device)

    # Instantiate Multiclass F1 score instance
    f1_score = MulticlassF1Score(num_classes=num_classes, average='macro').to(device)
    f1_score_mca = MulticlassF1Score(num_classes=num_classes, average=None).to(device)

    # ROC-AUC
    auroc = MulticlassAUROC(num_classes=num_classes, thresholds=None, average='macro').to(device)
    auroc_mca = MulticlassAUROC(num_classes=num_classes, average=None, thresholds=None).to(device)

    precision = MulticlassPrecision(num_classes=num_classes, average='macro').to(device)
    recall = MulticlassRecall(num_classes=num_classes, average='macro').to(device)

    encoded_buffer = ReplayBuffer()

    dataloader_A_iter = iter(itertools.cycle(dataloader_A))  # Cycle through dataloader_A
    dataloader_B_iter = iter(itertools.cycle(dataloader_B))  # Cycle through dataloader_B

    total_steps = max(len(dataloader_A), len(dataloader_B))
    # Loop through data loader data batches
    for _ in range(max(len(dataloader_A), len(dataloader_B))):  # Loop based on the longest dataloader
        try:
            data_A = next(dataloader_A_iter)
        except StopIteration:
            dataloader_A_iter = iter(dataloader_A)
            data_A = next(dataloader_A_iter)
        
        # Handle Dataloader B
        try:
            data_B, labels_B = next(dataloader_B_iter)
        except StopIteration:
            dataloader_B_iter = iter(dataloader_B)
            data_B, labels_B = next(dataloader_B_iter)

        # Send data to target device
        data_A = data_A.to(device)
        data_B, labels_B = data_B.to(device), labels_B.to(device)

        # ------------------ AutoEncoder Training ------------------
        optimizer_A.zero_grad()
        
        _, decoded = AE(data_A)

        # Calculate and accumulate AE training loss
        AE_loss = loss_fn_A(decoded, data_A)
        AE_train_loss += AE_loss.item() 

        # Forward pass through encoder and classifier with Dataset B (labeled)
        encoded_B, decoded_B = AE(data_B)  # Use encoder from autoencoder
        classification_output = classifier(encoded_B)

        cl_train_loss = loss_fn_C(classification_output, labels_B)
        
        cl_ae_loss = loss_fn_A(decoded_B, data_B)
        AE_train_loss += cl_ae_loss.item()

        total_loss = cl_train_loss * alpha + cl_ae_loss * beta + AE_loss * gamma 

        total_loss.backward()

        torch.nn.utils.clip_grad_norm_(AE.parameters(), max_norm=1.0)  # Clip gradients for AE
        optimizer_A.step()

        # ------------------ Classifier Training ------------------
        # Use encoded buffer and detach from computation graph for classifier-only training
        optimizer_B.zero_grad()
        encoded_B = encoded_buffer.push_and_pop(encoded_B)
        classification_output = classifier(encoded_B)

        loss = loss_fn_B(classification_output, labels_B)
        train_loss += loss.item() 

        loss.backward()

        #torch.nn.utils.clip_grad_norm_(classifier.parameters(), max_norm=1.0)  # Clip gradients for classifier
        optimizer_B.step()

        # Update accuracy and other metrics for this batch
        y_pred_class = torch.argmax(classification_output, dim=1)

        acc.update(y_pred_class, labels_B)
        acc_mca.update(classification_output, labels_B)
        f1_score.update(y_pred_class, labels_B)
        f1_score_mca.update(classification_output, labels_B)
        auroc.update(classification_output, labels_B)
        auroc_mca.update(classification_output, labels_B)
        precision.update(y_pred_class, labels_B)
        recall.update(y_pred_class, labels_B)

    # Adjust metrics to get average loss and accuracy per batch 
    AE_train_loss = AE_train_loss / total_steps

    train_loss = train_loss / total_steps

    train_acc = acc.compute().cpu().numpy()
    train_acc_mca = acc_mca.compute().cpu().numpy()

    train_f1_score = f1_score.compute().cpu().numpy()
    train_f1_score_mca = f1_score_mca.compute().cpu().numpy()

    train_auroc = auroc.compute().cpu().numpy()
    train_auroc_mca = auroc_mca.compute().cpu().numpy()

    train_precision = precision.compute().cpu().numpy()
    train_recall = recall.compute().cpu().numpy()

    acc.reset()
    acc_mca.reset()
    f1_score.reset()
    f1_score_mca.reset()
    auroc.reset()
    auroc_mca.reset()
    precision.reset()
    recall.reset()

    return AE_train_loss, train_loss, train_acc, train_acc_mca, train_f1_score, train_f1_score_mca, train_auroc, train_auroc_mca, train_precision, train_recall

def val_step(AE: torch.nn.Module, 
              classifier: torch.nn.Module, 
              dataloader_A: torch.utils.data.DataLoader, 
              dataloader_B: torch.utils.data.DataLoader, 
              loss_fn_A: torch.nn.Module,
              loss_fn_B: torch.nn.Module,
              device: torch.device,
              num_classes=6):
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
    classifier.eval()

    # Setup test loss and test accuracy values
    AE_val_loss= 0

    # Setup val loss and val accuracy values
    val_loss = 0

    # Instantiate Multiclass Accuracy instance
    acc = MulticlassAccuracy(num_classes=num_classes, average='macro').to(device)
    acc_mca = MulticlassAccuracy(num_classes=num_classes, average=None).to(device)

    # Instantiate Multiclass F1 score instance
    f1_score = MulticlassF1Score(num_classes=num_classes, average='macro').to(device)
    f1_score_mca = MulticlassF1Score(num_classes=num_classes, average=None).to(device)

    # ROC-AUC
    auroc = MulticlassAUROC(num_classes=num_classes, thresholds=None, average='macro').to(device)
    auroc_mca = MulticlassAUROC(num_classes=num_classes, average=None, thresholds=None).to(device)

    precision = MulticlassPrecision(num_classes=num_classes, average="macro").to(device)
    recall = MulticlassRecall(num_classes=num_classes, average="macro").to(device)

    # Turn on inference context manager
    with torch.inference_mode():
        # Loop through DataLoader batches
        dataloader_A_iter = iter(itertools.cycle(dataloader_A))  # Cycle through dataloader_A
        dataloader_B_iter = iter(itertools.cycle(dataloader_B))  # Cycle through dataloader_B
        total_steps = max(len(dataloader_A), len(dataloader_B))
        # Loop through data loader data batches
        for _ in range(max(len(dataloader_A), len(dataloader_B))):  # Loop based on the longest dataloader
            try:
                data_A = next(dataloader_A_iter)
            except StopIteration:
                dataloader_A_iter = iter(dataloader_A)
                data_A = next(dataloader_A_iter)
            
            # Handle Dataloader B
            try:
                data_B, labels_B = next(dataloader_B_iter)
            except StopIteration:
                dataloader_B_iter = iter(dataloader_B)
                data_B, labels_B = next(dataloader_B_iter)
            # Send data to target device
            data_A = data_A.to(device)

            data_B, labels_B = data_B.to(device), labels_B.to(device)

            # 1. Forward pass
            _, decoded = AE(data_A)

            # 2. Calculate and accumulate loss
            AE_loss = loss_fn_A(decoded, data_A)
            AE_val_loss += AE_loss.item()

            encoded_B, decoded_B = AE(data_B)  # Use encoder from autoencoder
            val_pred_logits = classifier(encoded_B)

            # 2. Calculate and accumulate loss
            loss = loss_fn_B(val_pred_logits, labels_B)
            val_loss += loss.item()

            cl_ae_loss = loss_fn_A(decoded_B, data_B)
            AE_val_loss += cl_ae_loss.item()

            # Calculate and accumulate accuracy
            val_pred_labels = torch.argmax(val_pred_logits, dim=1)

            # Update metrics for each batch
            acc.update(val_pred_labels, labels_B)
            acc_mca.update(val_pred_logits, labels_B)
            f1_score.update(val_pred_labels, labels_B)
            f1_score_mca.update(val_pred_logits, labels_B)
            auroc.update(val_pred_logits, labels_B)
            auroc_mca.update(val_pred_logits, labels_B)
            precision.update(val_pred_labels, labels_B)
            recall.update(val_pred_labels, labels_B)

        # Adjust metrics to get average loss and accuracy per batch 
        val_loss = val_loss / total_steps
        AE_val_loss = AE_val_loss / total_steps

        val_acc = acc.compute().cpu().numpy()
        val_acc_mca = acc_mca.compute().cpu().numpy()

        val_f1_score = f1_score.compute().cpu().numpy()
        val_f1_score_mca = f1_score_mca.compute().cpu().numpy()

        val_auroc = auroc.compute().cpu().numpy()
        val_auroc_mca = auroc_mca.compute().cpu().numpy()

        val_precision = precision.compute().cpu().numpy()
        val_recall = recall.compute().cpu().numpy()

        # Adjust metrics to get average loss and accuracy per batch 
        
    
    acc.reset()
    acc_mca.reset()
    f1_score.reset()
    f1_score_mca.reset()
    auroc.reset()
    auroc_mca.reset()
    precision.reset()
    recall.reset()
    return AE_val_loss, val_loss, val_acc, val_acc_mca, val_f1_score, val_f1_score_mca, val_auroc, val_auroc_mca, val_precision, val_recall

def test_step(AE: torch.nn.Module, 
              classifier: torch.nn.Module, 
              dataloader_A: torch.utils.data.DataLoader, 
              dataloader_B: torch.utils.data.DataLoader, 
              loss_fn_A: torch.nn.Module,
              loss_fn_B: torch.nn.Module,
              device: torch.device,
              num_classes=6):
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
    classifier.eval()

    # Setup test loss and test accuracy values
    AE_test_loss= 0

    # Setup test loss and test accuracy values
    test_loss = 0

    # Instantiate Multiclass Accuracy instance
    acc = MulticlassAccuracy(num_classes=num_classes, average='macro').to(device)
    acc_mca = MulticlassAccuracy(num_classes=num_classes, average=None).to(device)

    # Instantiate Multiclass F1 score instance
    f1_score = MulticlassF1Score(num_classes=num_classes, average='macro').to(device)
    f1_score_mca = MulticlassF1Score(num_classes=num_classes, average=None).to(device)

    # ROC-AUC
    auroc = MulticlassAUROC(num_classes=num_classes, thresholds=None, average='macro').to(device)
    auroc_mca = MulticlassAUROC(num_classes=num_classes, average=None, thresholds=None).to(device)

    precision = MulticlassPrecision(num_classes=num_classes, average="macro").to(device)
    recall = MulticlassRecall(num_classes=num_classes, average="macro").to(device)

    # Turn on inference context manager
    with torch.inference_mode():
        # Loop through DataLoader batches
        dataloader_A_iter = iter(itertools.cycle(dataloader_A))  # Cycle through dataloader_A
        dataloader_B_iter = iter(itertools.cycle(dataloader_B))  # Cycle through dataloader_B
        total_steps = max(len(dataloader_A), len(dataloader_B))
        # Loop through data loader data batches
        for _ in range(max(len(dataloader_A), len(dataloader_B))):  # Loop based on the longest dataloader
            try:
                data_A = next(dataloader_A_iter)
            except StopIteration:
                dataloader_A_iter = iter(dataloader_A)
                data_A = next(dataloader_A_iter)
            
            # Handle Dataloader B
            try:
                data_B, labels_B = next(dataloader_B_iter)
            except StopIteration:
                dataloader_B_iter = iter(dataloader_B)
                data_B, labels_B = next(dataloader_B_iter)
            # Send data to target device
            data_A = data_A.to(device)

            data_B, labels_B = data_B.to(device), labels_B.to(device)

            # 1. Forward pass
            _, decoded = AE(data_A)

            # 2. Calculate and accumulate loss
            AE_loss = loss_fn_A(decoded, data_A)
            AE_test_loss += AE_loss.item()

            encoded_B, decoded_B = AE(data_B)  # Use encoder from autoencoder
            test_pred_logits = classifier(encoded_B)

            # 2. Calculate and accumulate loss
            loss = loss_fn_B(test_pred_logits, labels_B)
            test_loss += loss.item()

            cl_ae_loss = loss_fn_A(decoded_B, data_B)
            AE_test_loss += cl_ae_loss.item()

            # Calculate and accumulate accuracy
            test_pred_labels = torch.argmax(test_pred_logits, dim=1)

            # Update metrics for each batch
            acc.update(test_pred_labels, labels_B)
            acc_mca.update(test_pred_logits, labels_B)
            f1_score.update(test_pred_labels, labels_B)
            f1_score_mca.update(test_pred_logits, labels_B)
            auroc.update(test_pred_logits, labels_B)
            auroc_mca.update(test_pred_logits, labels_B)
            precision.update(test_pred_labels, labels_B)
            recall.update(test_pred_labels, labels_B)

        # Adjust metrics to get average loss and accuracy per batch 
        test_loss = test_loss / total_steps
        AE_test_loss = AE_test_loss / total_steps

        test_acc = acc.compute().cpu().numpy()
        test_acc_mca = acc_mca.compute().cpu().numpy()

        test_f1_score = f1_score.compute().cpu().numpy()
        test_f1_score_mca = f1_score_mca.compute().cpu().numpy()

        test_auroc = auroc.compute().cpu().numpy()
        test_auroc_mca = auroc_mca.compute().cpu().numpy()

        test_precision = precision.compute().cpu().numpy()
        test_recall = recall.compute().cpu().numpy()

        # Adjust metrics to get average loss and accuracy per batch 
        
    
    acc.reset()
    acc_mca.reset()
    f1_score.reset()
    f1_score_mca.reset()
    auroc.reset()
    auroc_mca.reset()
    precision.reset()
    recall.reset()
    return AE_test_loss, test_loss, test_acc, test_acc_mca, test_f1_score, test_f1_score_mca, test_auroc, test_auroc_mca, test_precision, test_recall

def train(AE: torch.nn.Module, 
          classifier: torch.nn.Module, 
          train_dataloader_A: torch.utils.data.DataLoader,
          val_dataloader_A: torch.utils.data.DataLoader,
          test_dataloader_A: torch.utils.data.DataLoader, 
          train_dataloader_B: torch.utils.data.DataLoader, 
          test_dataloader_B: torch.utils.data.DataLoader,
          val_dataloader_B: torch.utils.data.DataLoader,
          optimizer_A: torch.optim.Optimizer,
          optimizer_B: torch.optim.Optimizer,
          loss_fn_A: torch.nn.Module,
          loss_fn_B: torch.nn.Module,
          loss_fn_C: torch.nn.Module,
          epochs: int,
          device: torch.device,
          scheduler_A,
          scheduler_B,
          training_progress_dir,
          model_weights_dir,
          fold_results,
          fold,
          num_classes=None,
          alpha=1,
          beta=10,
          gamma=5,
          AE_lr=5e-4,
          CL_lr=0.0001,
          patience=20,
          save_models: bool = True):
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

    csv_file_path = model_weights_dir / f"{fold}.csv"
    test_file_path = model_weights_dir / f"{fold}_test.csv"
    # Initialize CSV file with header if it doesn't exist
    fieldnames = ["Epoch", "AE_train_loss", "AE_val_loss", "train_loss", "train_acc", "train_acc_mca", "train_f1", "train_f1_mca", "train_auroc", "train_auroc_mca", "train_precision", "train_recall", "val_loss", "val_acc", "val_acc_mca", "val_f1", "val_f1_mca", "val_auroc", "val_auroc_mca", "val_precision", "val_recall"]
    if not os.path.isfile(csv_file_path):
        with open(csv_file_path, "w", newline='') as csvfile:
            csv_writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            csv_writer.writeheader()
    tst_fieldnames = ["Fold", "AE_test_loss", "test_loss", "test_acc", "test_acc_mca", "test_f1", "test_f1_mca", "test_auroc", "test_auroc_mca", "test_precision", "test_recall"]
    if not os.path.isfile(test_file_path):
        with open(test_file_path, "w", newline='') as tst_csvfile:
            tst_csv_writer = csv.DictWriter(tst_csvfile, fieldnames=tst_fieldnames)
            tst_csv_writer.writeheader()
            
    best_f1 = -np.inf
    best_epoch_cl = -1

    best_loss  = np.inf
    best_epoch_ae = -1
    
    # Make sure model on target device
    classifier.to(device)
    AE.to(device)

    # early_stopping_loss = EarlyStoppingLoss(patience=20, min_delta=0.0001)
    early_stopping_f1 = EarlyStoppingF1(patience=patience, min_delta=0.001)

    # Loop through training and testing steps for a number of epochs
    for epoch in tqdm(range(epochs)):

        AE_train_loss, train_loss, train_acc, train_acc_mca, train_f1, train_f1_mca, train_auroc, train_auroc_mca, train_precision, train_recall = train_step(AE=AE, 
                                                                                                                                classifier=classifier, 
                                                                                                                                dataloader_A=train_dataloader_A,
                                                                                                                                dataloader_B=train_dataloader_B, 
                                                                                                                                loss_fn_A=loss_fn_A,
                                                                                                                                loss_fn_B = loss_fn_B,
                                                                                                                                loss_fn_C=loss_fn_C,
                                                                                                                                optimizer_A=optimizer_A,
                                                                                                                                optimizer_B=optimizer_B,
                                                                                                                                device=device,
                                                                                                                                num_classes=num_classes,
                                                                                                                                alpha=alpha,
                                                                                                                                beta=beta,
                                                                                                                                gamma=gamma)

        AE_val_loss, val_loss, val_acc, val_acc_mca, val_f1, val_f1_mca, val_auroc, val_auroc_mca, val_precision, val_recall = val_step(AE=AE, 
                                                                                                                    classifier=classifier, 
                                                                                                                    dataloader_A=val_dataloader_A,
                                                                                                                    dataloader_B=val_dataloader_B, 
                                                                                                                    loss_fn_A=loss_fn_A,
                                                                                                                    loss_fn_B=loss_fn_B,
                                                                                                                    device=device,
                                                                                                                    num_classes=num_classes)
        
        scheduler_A.step(AE_val_loss)
        scheduler_B.step(val_f1)

        # Print out what's happening
        print(
          f"{sr_}Epoch: {epoch+1}\n"
          f"\t|AE_train_loss: {AE_train_loss:.4f} | AE_val_loss: {AE_val_loss:.4f}"
        )

        print(
          f"{sr_}Epoch: {epoch+1}\n"
          f"\t|train_loss: {train_loss:.4f}\n"
          f"\t|train_acc: {train_acc:.4f} | "
          f"train_f1: {train_f1:.4f} | "
          f"train_auroc: {train_auroc:.4f} | "
          f"train_precision: {train_precision:.4f} | "
          f"train_recall: {train_recall:.4f}\n"
          f"\t|val_loss: {val_loss:.4f} \n"
          f"\t|val_acc: {val_acc:.4f} | "
          f"val_f1: {val_f1:.4f} | "
          f"val_auroc: {val_auroc:.4f} | "
          f"val_precision: {val_precision:.4f} | "
          f"val_recall: {val_recall:.4f}"
        )

        # deep copy the model
        if AE_val_loss < best_loss:
            print(f"{c_}AE val Loss Improved ({best_loss:0.4f} ---> {AE_val_loss:0.4f})")
            best_loss = AE_val_loss
            best_epoch_ae = epoch+1
            
            # save validation output reconstruction for the current epoch
            display_random_images(data_loader=val_dataloader_A,
                                  AE=AE,
                                  file_recon=os.path.join(training_progress_dir, f"epoch{epoch + 1}_fold{fold}_val_recon.png"),
                                  display_real=False,
                                  device=device)
            print(f"{sr_}")
        
         # deep copy the model
        if val_f1 > best_f1:
            print(f"{c_}CL val F1 Improved ({best_f1:0.4f} ---> {val_f1:0.4f})")
            best_f1 = val_f1
            best_epoch_cl = epoch+1

            if save_models:
                # Save the model
                CL_save_filepath = f"CL_{epochs}_epochs_{fold}fold.pth"
                save_model(
                    model_state_dict=classifier.state_dict(),
                    target_dir=model_weights_dir,
                    model_name=CL_save_filepath,
                )

                # Save the model
                AE_save_filepath = f"AE_{epochs}_epochs_{fold}fold.pth"
                save_model(
                    model_state_dict=AE.state_dict(),
                    target_dir=model_weights_dir,
                    model_name=AE_save_filepath,
                )
            
            print(f"{sr_}")

        # Write the results to CSV file
        with open(csv_file_path, "a", newline='') as csvfile:
            csv_writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            row = {
                "Epoch": epoch + 1,
                "AE_train_loss": AE_train_loss,
                "AE_val_loss": AE_val_loss,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "train_acc_mca": train_acc_mca,
                "train_f1": train_f1,
                "train_f1_mca": train_f1_mca,
                "train_auroc": train_auroc,
                "train_auroc_mca": train_auroc_mca,
                "train_precision": train_precision,
                "train_recall": train_recall,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "val_acc_mca": val_acc_mca,
                "val_f1": val_f1,
                "val_f1_mca": val_f1_mca,
                "val_auroc": val_auroc,
                "val_auroc_mca": val_auroc_mca,
                "val_precision": val_precision,
                "val_recall": val_recall
            }
            csv_writer.writerow(row)

        if scheduler_A.get_last_lr()[-1] != AE_lr:
            print(f"{g_}AE learning rate decreased: {AE_lr} -> {scheduler_A.get_last_lr()[-1]}")
            AE_lr = scheduler_A.get_last_lr()[-1]
            print(f"{sr_}")

        if scheduler_B.get_last_lr()[-1] != CL_lr:
            print(f"{g_}CL learning rate decreased: {CL_lr} -> {scheduler_B.get_last_lr()[-1]}")
            CL_lr = scheduler_B.get_last_lr()[-1]
            alpha = lambda_schedule(epoch+1, 100, initial_lambda=1.0, final_lambda=10.0, growth_factor=1.0)
            print(f"alpha increased: {alpha}")
            print(f"{sr_}")
        # Check early stopping conditions
        early_stopping_f1(val_f1)

        #if early_stopping_loss.early_stop and early_stopping_f1.early_stop:
        if early_stopping_f1.early_stop:
            print(f"{w_}Early stopping at epoch {epoch+1}")
            break
    
    AE_test_loss, test_loss, test_acc, test_acc_mca, test_f1, test_f1_mca, test_auroc, test_auroc_mca, test_precision, test_recall = test_step(AE=AE, 
                                                                                                            classifier=classifier, 
                                                                                                            dataloader_A=test_dataloader_A,
                                                                                                            dataloader_B=test_dataloader_B, 
                                                                                                            loss_fn_A=loss_fn_A,
                                                                                                            loss_fn_B=loss_fn_B,
                                                                                                            device=device,
                                                                                                            num_classes=num_classes)
    display_random_images(data_loader=test_dataloader_A,
                                  AE=AE,
                                  file_recon=os.path.join(training_progress_dir, f"fold{fold}_test_recon.png"),
                                  display_real=False,
                                  device=device)
    fold_results["AE_test_loss"].append(AE_test_loss)
    fold_results["test_loss"].append(test_loss)
    fold_results["test_acc"].append(test_acc)
    fold_results["test_acc_mca"].append(test_acc_mca)
    fold_results["test_f1"].append(test_f1)
    fold_results["test_f1_mca"].append(test_f1_mca)
    fold_results["test_auroc"].append(test_auroc)
    fold_results["test_auroc_mca"].append(test_auroc_mca)
    fold_results["test_precision"].append(test_precision)
    fold_results["test_recall"].append(test_recall)

    # Write the results to CSV file
    with open(test_file_path, "a", newline='') as tst_csvfile:
        tst_csv_writer = csv.DictWriter(tst_csvfile, fieldnames=tst_fieldnames)
        row = {
            "Fold": fold,
            "AE_test_loss": AE_test_loss,
            "test_loss": test_loss,
            "test_acc": test_acc,
            "test_acc_mca": test_acc_mca,
            "test_f1": test_f1,
            "test_f1_mca": test_f1_mca,
            "test_auroc": test_auroc,
            "test_auroc_mca": test_auroc_mca,
            "test_precision": test_precision,
            "test_recall": test_recall
        }
        tst_csv_writer.writerow(row)

    print(f"{c_}[INFO] Final Val Results:")
    print(
          f"Best CL Epoch: {best_epoch_cl}\n"
          f"\t|Best CL F1: {best_f1:.4f}"
        )
    
    print(
          f"Best AE Epoch: {best_epoch_ae}\n"
          f"\t|Best AE Loss: {best_loss:.4f}"
        )
    print(f"{c_}[INFO] Final Test Results:")
    for metric, values in fold_results.items():
        if "mca" not in metric:
            print(f"{metric}: {values[-1]:.4f}")
    
    print(f"{sr_}")
