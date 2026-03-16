# Visualization helpers: curves, confusion matrix, embeddings, and gradient norms.

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from config import SEED, seed_everything, DEVICE, NUM_CLASSES

try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False

# Curves

def plot_curves(hist: dict, title: str, path: str):
    """Plot accuracy and loss curves side by side."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5))
    eps = range(1, len(hist["ta"]) + 1)
    a1.plot(eps, hist["ta"], label="Train"); a1.plot(eps, hist["va"], label="Val")
    a1.set_xlabel("Epoch"); a1.set_ylabel("Accuracy")
    a1.set_title(f"{title} - Accuracy"); a1.legend(); a1.grid(True)
    a2.plot(eps, hist["tl"], label="Train"); a2.plot(eps, hist["vl"], label="Val")
    a2.set_xlabel("Epoch"); a2.set_ylabel("Loss")
    a2.set_title(f"{title} - Loss"); a2.legend(); a2.grid(True)
    plt.tight_layout(); plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"    -> saved {path}")

# Confusion matrix

def plot_confusion_matrix(model, loader, class_names: list[str], title: str, path: str):
    """Compute and save confusion matrix."""
    model.eval()
    preds_all, labs_all = [], []
    with torch.no_grad():
        for imgs, labs in loader:
            out = model(imgs.to(DEVICE))
            if isinstance(out, tuple):
                out = out[0]
            preds_all.extend(out.argmax(1).cpu().numpy())
            labs_all.extend(labs.numpy())
    cm = confusion_matrix(labs_all, preds_all)
    fig, ax = plt.subplots(figsize=(16, 14))
    ConfusionMatrixDisplay(cm, display_labels=class_names).plot(
        ax=ax, xticks_rotation=90, cmap="Blues", values_format="d",
    )
    ax.set_title(title, fontsize=13)
    plt.tight_layout(); plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"    -> saved {path}")

# Embeddings

def plot_embeddings(feats: np.ndarray, labs: np.ndarray, class_names: list[str],
                    model_name: str, save_dir: str,
                    methods: tuple[str, ...] = ("PCA", "TSNE", "UMAP")):
    """PCA / t-SNE / UMAP scatter plots of feature embeddings."""
    seed_everything(SEED)
    nc   = len(class_names)
    cmap = plt.cm.get_cmap("nipy_spectral", nc)
    for method in methods:
        print(f"    Computing {method} ...")
        if method == "PCA":
            emb = PCA(n_components=2, random_state=SEED).fit_transform(feats)
        elif method == "TSNE":
            emb = TSNE(n_components=2, random_state=SEED, perplexity=30,
                       max_iter=1000).fit_transform(feats)
        elif method == "UMAP" and HAS_UMAP:
            emb = umap.UMAP(n_components=2, random_state=SEED, n_jobs=1).fit_transform(feats)
        else:
            continue
        fig, ax = plt.subplots(figsize=(12, 10))
        for c in range(nc):
            mask = labs == c
            ax.scatter(emb[mask, 0], emb[mask, 1], s=8, alpha=0.6,
                       label=class_names[c], color=cmap(c))
        ax.set_title(f"{model_name} - {method}")
        ax.legend(fontsize=5, ncol=3, markerscale=3)
        ax.grid(True, alpha=0.3)
        p = os.path.join(save_dir, f"{model_name}_{method.lower()}.png")
        plt.tight_layout(); plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
        print(f"    -> saved {p}")

# Gradient norms

def plot_grad_bars(gnorms: dict[str, float], model_name: str, strategy: str, path: str):
    """Bar chart of per-group average gradient norms."""
    agg: dict[str, list[float]] = {}
    for name, val in gnorms.items():
        top = name.split(".")[0]
        agg.setdefault(top, []).append(val)
    agg_mean = {k: float(np.mean(v)) for k, v in agg.items()}
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(range(len(agg_mean)), list(agg_mean.values()), tick_label=list(agg_mean.keys()))
    ax.set_ylabel("Avg Gradient Norm")
    ax.set_title(f"{model_name} - {strategy}")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout(); plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"    -> saved {path}")

# Sensitivity bars (Bonus)

def plot_sensitivity(sens: dict[str, float], selected: list, model_name: str, path: str):
    """Horizontal bar chart: green = auto-selected layers."""
    sel_names = {s[0] for s in selected}
    names  = list(sens.keys())
    scores = [sens[n] for n in names]
    colors = ["#2ecc71" if n in sel_names else "#bdc3c7" for n in names]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(names, scores, color=colors)
    ax.set_xlabel("Sensitivity Score")
    ax.set_title(f"{model_name} - Layer Sensitivity (green = auto-selected)")
    ax.invert_yaxis()
    plt.tight_layout(); plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"    -> saved {path}")
