import torch
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import inspect

import torch

from sklearn.metrics import pairwise_distances
import numpy as np

def compute_cluster_metrics(latents, labels):
    unique_classes = np.unique(labels)

    intra_dists = []
    class_means = []

    for c in unique_classes:
        class_points = latents[labels == c]

        if len(class_points) < 2:
            continue

        dists = pairwise_distances(class_points)
        intra_dists.append(np.mean(dists))
        class_means.append(np.mean(class_points, axis=0))

    D_intra = np.mean(intra_dists)

    class_means = np.array(class_means)
    inter_dists = pairwise_distances(class_means)
    D_inter = np.mean(inter_dists[np.triu_indices(len(class_means), k=1)])

    score = D_inter / (D_intra + 1e-8)

    return D_intra, D_inter, score

# Hyperparameters
def plot_tsne(
    AE,
    dataloader,
    device,
    num_samples=2000,
    perplexity=30,
    n_iter=1000,
    seed=42,
    save_path="tsne_plot.png",
    show: bool = False,
):
    AE.eval()

    all_latents = []
    all_labels = []

    with torch.no_grad():
        for data, labels in dataloader:
            data = data.to(device)
            encoded, _ = AE(data)
            all_latents.append(encoded.cpu())
            all_labels.append(labels.cpu())

            if len(torch.cat(all_latents)) > num_samples:
                break

    latents = torch.cat(all_latents)[:num_samples].numpy()
    labels = torch.cat(all_labels)[:num_samples].numpy()

    # ---- Cluster Quality Metrics ----
    D_intra, D_inter, score = compute_cluster_metrics(latents, labels)

    print("\n[CLUSTER METRICS]")
    print(f"Intra-class distance: {D_intra:.4f}")
    print(f"Inter-class distance: {D_inter:.4f}")
    print(f"Cluster score (inter/intra): {score:.4f}")

    print("Running t-SNE...")
    # sklearn renamed n_iter -> max_iter; prefer max_iter when available to avoid warnings.
    sig = None
    try:
        sig = inspect.signature(TSNE)
    except Exception:
        sig = None
    if sig is not None and "max_iter" in sig.parameters:
        tsne = TSNE(n_components=2, perplexity=perplexity, max_iter=n_iter, random_state=seed)
    else:
        tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter, random_state=seed)
    tsne_result = tsne.fit_transform(latents)

    # Plot
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=tsne_result[:, 0], y=tsne_result[:, 1], hue=labels, palette="tab10", s=60, alpha=0.8)
    plt.legend(title="Class", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, format='svg', bbox_inches='tight', pad_inches=0.05)
    print(f"[✔] t-SNE plot saved to {save_path}")
    if show:
        plt.show()
    else:
        plt.close()
