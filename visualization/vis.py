import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import os
import ast

# Plot confusion matrix
from torchmetrics import ConfusionMatrix
#from mlxtend.plotting import plot_confusion_matrix
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score
# Import tqdm for progress bar
from tqdm.auto import tqdm

import torch

#plt.style.use('ggplot')  # Try styles like 'seaborn', 'fivethirtyeight', etc.

def plot_configs():
    # Adjust global font size, line width, and figure size
    plt.rcParams.update({
        'font.size': 14,        # Increase font size
        'axes.titlesize': 16,   # Title font size
        'axes.labelsize': 14,   # Axis label font size
        'lines.linewidth': 2.5, # Line width for plot lines
        'xtick.labelsize': 12,  # X-axis tick label font size
        'ytick.labelsize': 12,  # Y-axis tick label font size
        'legend.fontsize': 12,  # Legend font size
    })

# Sebastian Raschka 2014-2024
# mlxtend Machine Learning Library Extensions
# Author: Sebastian Raschka <sebastianraschka.com>
#
# A function for plotting a confusion matrix.
# License: BSD 3 clause
def plot_confusion_matrix(
    conf_mat,
    hide_spines=False,
    hide_ticks=False,
    figsize=None,
    cmap=None,
    colorbar=False,
    show_absolute=True,
    show_normed=False,
    norm_colormap=None,
    class_names=None,
    figure=None,
    axis=None,
    fontcolor_threshold=0.5,
):
    """Plot a confusion matrix via matplotlib.

    Parameters
    -----------
    conf_mat : array-like, shape = [n_classes, n_classes]
        Confusion matrix from evaluate.confusion matrix.

    hide_spines : bool (default: False)
        Hides axis spines if True.

    hide_ticks : bool (default: False)
        Hides axis ticks if True

    figsize : tuple (default: (2.5, 2.5))
        Height and width of the figure

    cmap : matplotlib colormap (default: `None`)
        Uses matplotlib.pyplot.cm.Blues if `None`

    colorbar : bool (default: False)
        Shows a colorbar if True

    show_absolute : bool (default: True)
        Shows absolute confusion matrix coefficients if True.
        At least one of  `show_absolute` or `show_normed`
        must be True.

    show_normed : bool (default: False)
        Shows normed confusion matrix coefficients if True.
        The normed confusion matrix coefficients give the
        proportion of training examples per class that are
        assigned the correct label.
        At least one of  `show_absolute` or `show_normed`
        must be True.

    norm_colormap : bool (default: False)
        Matplotlib color normalization object to normalize the
        color scale, e.g., `matplotlib.colors.LogNorm()`.

    class_names : array-like, shape = [n_classes] (default: None)
        List of class names.
        If not `None`, ticks will be set to these values.

    figure : None or Matplotlib figure  (default: None)
        If None will create a new figure.

    axis : None or Matplotlib figure axis (default: None)
        If None will create a new axis.

    fontcolor_threshold : Float (default: 0.5)
        Sets a threshold for choosing black and white font colors
        for the cells. By default all values larger than 0.5 times
        the maximum cell value are converted to white, and everything
        equal or smaller than 0.5 times the maximum cell value are converted
        to black.

    Returns
    -----------
    fig, ax : matplotlib.pyplot subplot objects
        Figure and axis elements of the subplot.

    Examples
    -----------
    For usage examples, please see
    https://rasbt.github.io/mlxtend/user_guide/plotting/plot_confusion_matrix/

    """
    if not (show_absolute or show_normed):
        raise AssertionError("Both show_absolute and show_normed are False")
    if class_names is not None and len(class_names) != len(conf_mat):
        raise AssertionError(
            "len(class_names) should be equal to number of" "classes in the dataset"
        )

    total_samples = conf_mat.sum(axis=1)[:, np.newaxis]
    normed_conf_mat = conf_mat.astype("float") / total_samples

    if figure is None and axis is None:
        fig, ax = plt.subplots(figsize=figsize)
    elif axis is None:
        fig = figure
        ax = fig.add_subplot(1, 1, 1)
    else:
        fig, ax = figure, axis

    ax.grid(False)
    if cmap is None:
        cmap = plt.cm.Blues

    if figsize is None:
        figsize = (len(conf_mat) * 1.25, len(conf_mat) * 1.25)

    if show_normed:
        matshow = ax.matshow(normed_conf_mat, cmap=cmap, norm=norm_colormap)
    else:
        matshow = ax.matshow(conf_mat, cmap=cmap, norm=norm_colormap)

    if colorbar:
        fig.colorbar(matshow)

    for i in range(conf_mat.shape[0]):
        for j in range(conf_mat.shape[1]):
            cell_text = ""
            if show_absolute:
                num = conf_mat[i, j].astype(np.int64)
                cell_text += format(num, "d")
                if show_normed:
                    cell_text += "\n" + "("
                    cell_text += format(normed_conf_mat[i, j], ".2f") + ")"
            else:
                cell_text += format(normed_conf_mat[i, j], ".2f")

            if show_normed:
                ax.text(
                    x=j,
                    y=i,
                    s=cell_text,
                    va="center",
                    ha="center",
                    color=(
                        "white"
                        if normed_conf_mat[i, j] > 1 * fontcolor_threshold
                        else "black"
                    ),
                )
            else:
                ax.text(
                    x=j,
                    y=i,
                    s=cell_text,
                    va="center",
                    ha="center",
                    color=(
                        "white"
                        if conf_mat[i, j] > np.max(conf_mat) * fontcolor_threshold
                        else "black"
                    ),
                )
    if class_names is not None:
        tick_marks = np.arange(len(class_names))
        plt.xticks(
            tick_marks, class_names, rotation=45, ha="right", rotation_mode="anchor"
        )
        plt.yticks(tick_marks, class_names)

    if hide_spines:
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
    ax.yaxis.set_ticks_position("left")
    ax.xaxis.set_ticks_position("bottom")
    if hide_ticks:
        ax.axes.get_yaxis().set_ticks([])
        ax.axes.get_xaxis().set_ticks([])

    #plt.xlabel("predicted label")
    #plt.ylabel("true label")
    return fig, ax

def plot_conf_matrix(AE, 
                     classifier,
                     class_names,
                     device,
                     l_test_dataloader,
                     colorbar=False):
    # Set the model to evaluation mode
    AE.eval()
    classifier.eval()

    # 1. Make predictions with trained model
    y_preds = []
    test_data_target = []

    num_classes=len(class_names)

    test_acc, test_f1_score = 0, 0
    # calculates statistic for each label
    test_acc_mca = np.zeros(num_classes)
    test_f1_score_mca = np.zeros(num_classes)

    # Instantiate Multiclass Accuracy instance
    acc = MulticlassAccuracy(num_classes=num_classes, average="macro").to(device)
    acc_mca = MulticlassAccuracy(num_classes=num_classes, average=None).to(device)

    # Instantiate Multiclass F1 score instance
    f1_score = MulticlassF1Score(num_classes=num_classes, average="macro").to(device)
    f1_score_mca = MulticlassF1Score(num_classes=num_classes, average=None).to(device)

    with torch.inference_mode():
        for X, y in tqdm(l_test_dataloader, desc="Making predictions"):
            # Send data and targets to target device
            X, y = X.to(device), y.to(device)
            # Do the forward pass
            # 1. Forward pass
            encoded_B, _ = AE(X)  # Use encoder from autoencoder
            y_logit = classifier(encoded_B)

            # Turn predictions from logits -> prediction probabilities -> predictions labels
            y_pred = torch.argmax(y_logit, dim=1) # note: perform softmax on the "logits" dimension, not "batch" dimension (in this case we have a batch size of 32, so can perform on dim=1)
            # Put predictions on CPU for evaluation
            y_preds.append(y_pred.cpu())
            test_data_target.append(y.cpu())

            # Update metrics for each batch
            acc.update(y_pred, y)
            acc_mca.update(y_logit, y)
            f1_score.update(y_pred, y)
            f1_score_mca.update(y_logit, y)

        test_acc = acc.compute().cpu().numpy()
        test_acc_mca = acc_mca.compute().cpu().numpy()
        test_f1_score = f1_score.compute().cpu().numpy()
        test_f1_score_mca = f1_score_mca.compute().cpu().numpy()

    print(
            f"\ttest_acc: {test_acc:.4f} | "
            f"test_acc_mca: {np.round(test_acc_mca, decimals=4)} | "
            f"test_f1: {test_f1_score:.4f} | "
            f"test_f1_mca: {np.round(test_f1_score_mca, decimals=4)}"
            )

    # Concatenate list of predictions into a tensor
    y_pred_tensor = torch.cat(y_preds)
    test_data_target_tensor = torch.cat(test_data_target)

    # 2. Setup confusion matrix instance and compare predictions to targets
    confmat = ConfusionMatrix(num_classes=num_classes, task='multiclass')
    confmat_tensor = confmat(preds=y_pred_tensor,
                            target=test_data_target_tensor)
    
    # 3. Plot the confusion matrix
    fig, ax = plot_confusion_matrix(
        conf_mat=confmat_tensor.numpy(), # matplotlib likes working with NumPy
        class_names=class_names, # turn the row and column labels into class names
        show_absolute=False,
        show_normed=True,
        colorbar=colorbar
    )

    # Adjust font and line styles
    #plt.setp(ax.texts, fontsize=14, fontweight='bold')  # Make text inside cells bold and larger
    #ax.xaxis.label.set_size(16)
    #ax.xaxis.label.set_weight('bold')
    #ax.yaxis.label.set_size(16)
    #ax.yaxis.label.set_weight('bold')
    #plt.xlabel('Predicted Label', fontsize=16, fontweight='bold')
    #plt.ylabel('True Label', fontsize=16, fontweight='bold')

    plt.show()

def print_max_metrics(data, class_names, loss=True):
    # Find the index of the maximum F1 score
    max_f1_index = data['val_f1'].idxmax()

    # Retrieve the row corresponding to this index
    max_f1_row = data.loc[max_f1_index]

    # Print the maximum F1 score and other relevant metrics in formatted strings
    print(f"Maximum F1 Score: {max_f1_row['val_f1']*100:.2f}%")
    print(f"Accuracy at max F1: {max_f1_row['val_acc']*100:.2f}%")
    print(f"Precision at max F1: {max_f1_row['val_precision']*100:.2f}%")
    print(f"Recall at max F1: {max_f1_row['val_recall']*100:.2f}%")
    print(f"Epoch at max F1: {int(max_f1_row['Epoch'])}")  # If 'epoch' is an integer
    print('---------------------------------------------------------\n')

    # Assuming MCA values are stored as a space-separated string with brackets in a column 'mca_values'
    # Define the MCA metrics
    mca_metrics = ['val_acc_mca', 'val_f1_mca', 'val_auroc_mca']

    for metric in mca_metrics:
        mca_values_str = max_f1_row[metric].strip('[]')  # Remove brackets
        mca_values = list(map(float, mca_values_str.split()))  # Split and convert to float

        # Format MCA values as percentages
        mca_values = [f"{value * 100:.2f} %" for value in mca_values]

        # Create a DataFrame for a table-like representation
        mca_df = pd.DataFrame({
            'Class Name': class_names,
            'MCA Value': mca_values
        })

        # Print the table
        print(metric + " Values for Each Class at Max F1 Score:")
        print(mca_df.to_string(index=False))
        print('---------------------------------------------------------\n')

    if loss:
        # Create a dictionary of max values
        min_loss = data['AE_val_loss'].min()

        print(f"Minimum AE Evaluation Loss: {min_loss:.4f}")

def fix_list_string(s):
    if not isinstance(s, str):
        return s
    # Remove brackets and strip
    s = s.strip()
    if s.startswith('[') and s.endswith(']'):
        inner = s[1:-1].strip()
        # Replace spaces between numbers by commas, e.g. "0.1 0.2" -> "0.1,0.2"
        inner_fixed = ",".join(inner.split())
        fixed = f"[{inner_fixed}]"
        return fixed
    return s

def summarize_best_fold_results(fold_data, include_mca=True):
    scalar_metrics = ['val_f1', 'val_acc', 'val_precision', 'val_recall']
    all_best_metrics = []
    all_mca_per_class = []  # list of lists
    best_epochs = []        # to store epoch numbers of best models

    for fold_idx, df in enumerate(fold_data):
        best_row = df.loc[df['val_f1'].idxmax()]
        metrics = {metric: best_row[metric] for metric in scalar_metrics}
        # Support both legacy 'epoch' and current 'Epoch' column names.
        if 'epoch' in best_row:
            best_epoch = int(best_row['epoch'])
        elif 'Epoch' in best_row:
            best_epoch = int(best_row['Epoch'])
        else:
            best_epoch = None
        metrics['epoch'] = best_epoch
        best_epochs.append(best_epoch)

        if include_mca:
            f1_mca_list = best_row['val_f1_mca']
            if isinstance(f1_mca_list, list):
                metrics['val_f1_mca_avg'] = np.mean(f1_mca_list)
                all_mca_per_class.append(f1_mca_list)

        all_best_metrics.append(metrics)

    best_df = pd.DataFrame(all_best_metrics)

    # Compute mean and std of scalar metrics
    summary = {
        'mean': best_df.drop(columns=['epoch']).mean(),
        'std': best_df.drop(columns=['epoch']).std()
    }

    print("=== Best-per-Fold Validation Results ===")
    print(best_df.to_string(index=False))

    print("\n=== Final Report (Mean ± Std over 5 folds) ===")
    for metric in best_df.columns:
        if metric == 'epoch':
            continue
        mean = summary['mean'][metric]
        std = summary['std'][metric]
        print(f"{metric}: {mean:.4f} ± {std:.4f}")

    print("\n=== Best Epochs per Fold ===")
    for i, epoch in enumerate(best_epochs):
        print(f"Fold {i}: Epoch {epoch}")

    # === NEW: Per-Class Macro F1 Summary ===
    if include_mca and all_mca_per_class:
        mca_array = np.array(all_mca_per_class)

        print("\n=== Per-Class F1 Score (val_f1_mca) over 5 folds ===")
        for class_idx in range(mca_array.shape[1]):
            class_scores = mca_array[:, class_idx]
            mean = np.mean(class_scores)
            std = np.std(class_scores)
            print(f"Class {class_idx}: {mean:.4f} ± {std:.4f}")

        return summary, mca_array, best_epochs

    return summary, best_epochs

def summarize_test_results(fold_data):
    scalar_metrics = ['test_acc', 'test_f1', 'test_precision', 'test_recall']
    mca_metrics = ['test_f1_mca', 'test_acc_mca', 'test_auroc_mca']

    # Combine all fold rows into a single DataFrame
    df_all = pd.concat(fold_data, ignore_index=True)

    print("=== Test Set Results (Per Fold) ===")
    print(df_all[scalar_metrics + mca_metrics].to_string(index=False))

    # Best fold (by highest test_f1)
    best_fold_idx = df_all['test_f1'].idxmax()
    best_fold = df_all.loc[best_fold_idx]
    print(f"\n=== Best Fold: {int(best_fold['Fold'])} (based on highest test_f1 = {best_fold['test_f1']:.4f}) ===")
    print(best_fold[scalar_metrics + mca_metrics].to_string())

    print("\n=== Scalar Metrics: Mean ± Std over 5 Folds ===")
    for metric in scalar_metrics:
        mean = df_all[metric].mean()
        std = df_all[metric].std()
        print(f"{metric}: {mean:.4f} ± {std:.4f}")

    print("\n=== Per-Class MCA Metrics: Mean ± Std ===")
    for mca_metric in mca_metrics:
        # Extract list of lists for this MCA metric
        mca_lists = df_all[mca_metric].tolist()  # shape: (5, num_classes)
        mca_array = np.array(mca_lists)

        print(f"\n{mca_metric}:")
        for class_idx in range(mca_array.shape[1]):
            class_scores = mca_array[:, class_idx]
            mean = np.mean(class_scores)
            std = np.std(class_scores)
            print(f"  Class {class_idx}: {mean:.4f} ± {std:.4f}")

    return df_all

########################################################################################################################################
def plot_AE_Loss_CV_all_folds(
    all_fold_data,
    metric_name,
    ylabel,
    save_path=None,
    filename="AE_PLOT_all_folds"
): 
    # Set up the plot
    plt.figure(figsize=(10, 6))

    # Color maps for folds
    cmap_train = plt.get_cmap("Blues")
    cmap_val = plt.get_cmap("Reds")
    num_folds = len(all_fold_data)

    for i, df in enumerate(all_fold_data):
        # Plot raw training loss per epoch
        plt.plot(
            df['Epoch'], df['AE_train_loss'],
            label=f"Train Fold {i+1}",
            color=cmap_train(0.3 + 0.7 * i / max(1, num_folds - 1)),
            linestyle='-',
            marker='o',
            markersize=4,
            alpha=0.8
        )
        # Plot raw validation loss per epoch
        plt.plot(
            df['Epoch'], df['AE_val_loss'],
            label=f"Val Fold {i+1}",
            color=cmap_val(0.3 + 0.7 * i / max(1, num_folds - 1)),
            linestyle='--',
            marker='x',
            markersize=4,
            alpha=0.8
        )

    plt.xlabel("Epoch", fontweight='bold')
    plt.ylabel(ylabel, fontweight='bold')
    # plt.title(f"{metric_name} per Fold", fontweight='bold')
    plt.legend(ncol=2, fontsize=9)
    plt.grid(True)
    plt.tight_layout()

    # Save the plot if requested
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        svg_path = os.path.join(save_path, f"{filename}.svg")
        plt.savefig(svg_path, format='svg', bbox_inches='tight')  # no dpi for SVG
        print(f"Plot saved to:\n  {svg_path}")

    plt.show()

def plot_metric_CV_all_folds(
    all_fold_data,
    metric_name,
    ylabel,
    save_path=None,
    filename="metric_plot_all_folds"
):
    """
    Plots metric curves for each CV fold separately (no interpolation).
    Useful for visualizing variation across folds.
    """

    plt.figure(figsize=(10, 6))
    plt.ylim(0, 110)

    cmap_train = plt.get_cmap("Blues")
    cmap_val = plt.get_cmap("Reds")
    num_folds = len(all_fold_data)

    for i, df in enumerate(all_fold_data):
        # Scale to percentage
        train_vals = df[f'train_{metric_name}'] * 100
        val_vals = df[f'val_{metric_name}'] * 100

        plt.plot(
            df['Epoch'], train_vals,
            label=f"Train Fold {i+1}",
            color=cmap_train(0.3 + 0.7 * i / max(1, num_folds - 1)),
            linestyle='-',
            marker='o',
            markersize=4,
            alpha=0.8
        )
        plt.plot(
            df['Epoch'], val_vals,
            label=f"Val Fold {i+1}",
            color=cmap_val(0.3 + 0.7 * i / max(1, num_folds - 1)),
            linestyle='--',
            marker='x',
            markersize=4,
            alpha=0.8
        )

    plt.xlabel("Epoch", fontweight='bold')
    plt.ylabel(ylabel, fontweight='bold')
    # plt.title(f"{metric_name.capitalize()} per Fold", fontweight='bold')
    plt.legend(ncol=2, fontsize=9)
    plt.grid(True)
    plt.tight_layout()

    if save_path:
        os.makedirs(save_path, exist_ok=True)
        svg_path = os.path.join(save_path, f"{filename}.svg")
        plt.savefig(svg_path, format='svg', bbox_inches='tight')  # no dpi for SVG
        print(f"Plot saved to:\n  {svg_path}")

    plt.show()

def plot_precision_recall_CV_all_folds(
    all_fold_data,
    save_path=None,
    filename="precision_recall_plot_all_folds"
):
    """
    Plots train/val precision and recall curves for each CV fold individually (no interpolation, no averaging).
    """

    plt.figure(figsize=(10, 6))
    plt.ylim(0, 1.1)

    num_folds = len(all_fold_data)

    cmap_precision_train = plt.get_cmap("Blues")
    cmap_precision_val = plt.get_cmap("Reds")
    cmap_recall_train = plt.get_cmap("Greens")
    cmap_recall_val = plt.get_cmap("Oranges")

    for i, df in enumerate(all_fold_data):
        alpha = 0.3 + 0.7 * i / max(1, num_folds - 1)

        plt.plot(
            df['Epoch'], df['train_precision'],
            label=f'Train Prec Fold {i+1}',
            color=cmap_precision_train(alpha),
            linestyle='-',
            marker='o',
            markersize=4,
            alpha=0.8
        )
        plt.plot(
            df['Epoch'], df['val_precision'],
            label=f'Val Prec Fold {i+1}',
            color=cmap_precision_val(alpha),
            linestyle='--',
            marker='x',
            markersize=4,
            alpha=0.8
        )

        plt.plot(
            df['Epoch'], df['train_recall'],
            label=f'Train Rec Fold {i+1}',
            color=cmap_recall_train(alpha),
            linestyle='-',
            marker='o',
            markersize=4,
            alpha=0.8
        )
        plt.plot(
            df['Epoch'], df['val_recall'],
            label=f'Val Rec Fold {i+1}',
            color=cmap_recall_val(alpha),
            linestyle='--',
            marker='x',
            markersize=4,
            alpha=0.8
        )

    plt.xlabel("Epoch", fontweight='bold')
    plt.ylabel("Score", fontweight='bold')
    # plt.title("Precision and Recall per Fold", fontweight='bold')
    plt.legend(ncol=2, fontsize=9)
    plt.grid(True)
    plt.tight_layout()

    if save_path:
        os.makedirs(save_path, exist_ok=True)
        svg_path = os.path.join(save_path, f"{filename}.svg")
        plt.savefig(svg_path, format='svg', bbox_inches='tight')  # no dpi for SVG
        print(f"Plot saved to:\n  {svg_path}")

    plt.show()

def plot_mca_metric_CV_all_folds_for_classes(
    all_fold_data,
    metric_name,
    ylabel,
    title,
    class_names,
    save_dir=None
):
    """
    Plots train/val per-class metric curves for each fold (no interpolation).
    One plot per class showing all fold curves individually.
    """

    num_classes = len(class_names)
    num_folds = len(all_fold_data)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    cmap_train = plt.get_cmap("Blues")
    cmap_val = plt.get_cmap("Reds")

    for cls_idx in range(num_classes):
        plt.figure(figsize=(10, 6))
        plt.ylim(0, 1.1)

        for i, df in enumerate(all_fold_data):
            epochs = df['Epoch'].values

            # Convert stringified arrays to numpy arrays, extract class index
            train_vals = np.array(
                df[f'train_{metric_name}']
                .apply(lambda x: np.fromstring(x.strip("[]"), sep=' '))
                .apply(lambda arr: arr[cls_idx])
            )
            val_vals = np.array(
                df[f'val_{metric_name}']
                .apply(lambda arr: arr[cls_idx])
            )

            train_color = cmap_train(0.3 + 0.7 * i / max(1, num_folds - 1))
            val_color = cmap_val(0.3 + 0.7 * i / max(1, num_folds - 1))

            plt.plot(
                epochs, train_vals,
                label=f'Train Fold {i+1}',
                color=train_color,
                linestyle='-',
                marker='o',
                markersize=4,
                alpha=0.8
            )
            plt.plot(
                epochs, val_vals,
                label=f'Val Fold {i+1}',
                color=val_color,
                linestyle='--',
                marker='x',
                markersize=4,
                alpha=0.8
            )

        plt.xlabel('Epoch', fontweight='bold')
        plt.ylabel(ylabel, fontweight='bold')
        # plt.title(f'{title} - Class {class_names[cls_idx]}', fontweight='bold')
        plt.legend(ncol=2, fontsize=9)
        plt.grid(True)
        plt.tight_layout()

        if save_dir:
            save_path = os.path.join(save_dir, f'{metric_name}_class_{class_names[cls_idx]}_all_folds.svg')
            plt.savefig(save_path, format='svg', bbox_inches='tight')
            print(f"Saved: {save_path}")

        plt.show()
