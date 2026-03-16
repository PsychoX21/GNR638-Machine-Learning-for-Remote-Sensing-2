# Scenario 5: Layer-Wise Feature Probing

import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
from torchvision import datasets
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

from config import (
    seed_everything, SEED, DEVICE, RESULTS_DIR,
    NUM_CLASSES, BATCH_SIZE, NUM_WORKERS,
)
from data import create_train_val_split, get_transforms, get_class_names
from models import create_model, get_probe_layers, print_model_info
from train import extract_at_layer
from config import TRAIN_DIR, VAL_DIR

DEPTHS = ["early", "middle", "final"]


def run(models: list[str]):
    """Run Scenario 5 for given models."""
    create_train_val_split()
    class_names = get_class_names()
    out_dir = os.path.join(RESULTS_DIR, "scenario5")
    os.makedirs(out_dir, exist_ok=True)
    results: dict[str, dict] = {}
    pca_data: dict[tuple, tuple] = {}

    print("\n" + "=" * 70)
    print("SCENARIO 5: LAYER-WISE FEATURE PROBING")
    print("=" * 70)

    # Fixed PCA subset (same for all models/layers)
    seed_everything(SEED)
    pca_ds_base = datasets.ImageFolder(VAL_DIR, transform=get_transforms("resnet50", augment=False))
    pca_targets = np.array(pca_ds_base.targets)
    pca_idx = []
    for c in range(NUM_CLASSES):
        ci = np.where(pca_targets == c)[0][:30]
        pca_idx.extend(ci.tolist())
    print(f"PCA subset: {len(pca_idx)} samples ({NUM_CLASSES} classes x 30 samples)")

    for mname in models:
        print(f"\n>>> {mname}")
        seed_everything(SEED)

        common_dl = dict(batch_size=BATCH_SIZE, shuffle=False,
                         num_workers=NUM_WORKERS, pin_memory=True,
                         persistent_workers=NUM_WORKERS > 0)
        train_ds = datasets.ImageFolder(TRAIN_DIR, transform=get_transforms(mname, augment=False))
        val_ds   = datasets.ImageFolder(VAL_DIR,   transform=get_transforms(mname, augment=False))
        train_ld = DataLoader(train_ds, **common_dl)
        val_ld   = DataLoader(val_ds,   **common_dl)

        pca_ds   = datasets.ImageFolder(VAL_DIR, transform=get_transforms(mname, augment=False))
        pca_ld   = DataLoader(Subset(pca_ds, pca_idx), **common_dl)

        model = create_model(mname).to(DEVICE)
        print_model_info(model, mname)
        model.eval()
        probe_layers = get_probe_layers(model, mname)
        results[mname] = {}

        for depth in DEPTHS:
            lmod = probe_layers[depth]
            print(f"  [{depth}] Extracting features ...")

            train_feats, train_labs = extract_at_layer(model, lmod, train_ld)
            val_feats,   val_labs   = extract_at_layer(model, lmod, val_ld)
            norms = np.linalg.norm(val_feats, axis=1)
            print(f"    dim={val_feats.shape[1]}, norm={norms.mean():.2f} +/- {norms.std():.2f}")

            clf = LogisticRegression(
                max_iter=5000, random_state=SEED, C=1.0,
                solver="lbfgs",
            )
            clf.fit(train_feats, train_labs)
            val_acc = clf.score(val_feats, val_labs)
            print(f"    Val probe accuracy: {val_acc:.4f}")

            results[mname][depth] = {
                "acc":    val_acc,
                "dim":    int(val_feats.shape[1]),
                "norm_m": float(norms.mean()),
                "norm_s": float(norms.std()),
            }

            pca_f, pca_l = extract_at_layer(model, lmod, pca_ld)
            pca_2d = PCA(n_components=2, random_state=SEED).fit_transform(pca_f)
            pca_data[(mname, depth)] = (pca_2d, pca_l)

    # Accuracy vs depth
    fig, ax = plt.subplots(figsize=(10, 6))
    for m in models:
        ax.plot(DEPTHS, [results[m][d]["acc"] for d in DEPTHS], "o-", label=m, ms=8)
    ax.set_xlabel("Depth"); ax.set_ylabel("Probe Val Accuracy")
    ax.set_title("Layer-Wise Probe Accuracy vs Network Depth")
    ax.legend(); ax.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "acc_vs_depth.png"), dpi=150); plt.close()

    # Feature norms
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(DEPTHS)); w = 0.15
    for i, m in enumerate(models):
        means = [results[m][d]["norm_m"] for d in DEPTHS]
        stds  = [results[m][d]["norm_s"] for d in DEPTHS]
        ax.bar(x + i * w, means, w, yerr=stds, label=m, capsize=3)
    ax.set_xticks(x + w * (len(models) - 1) / 2); ax.set_xticklabels(DEPTHS)
    ax.set_ylabel("Feature Norm"); ax.set_title("Feature Norms Across Depths")
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "norms.png"), dpi=150); plt.close()

    # PCA grid
    nc = len(class_names)
    cmap_pca = plt.cm.get_cmap("nipy_spectral", nc)
    fig, axes = plt.subplots(len(models), 3, figsize=(18, 5 * len(models)))
    if len(models) == 1:
        axes = [axes]
    for i, m in enumerate(models):
        for j, d in enumerate(DEPTHS):
            ax  = axes[i][j]
            emb, lb = pca_data[(m, d)]
            for c in range(nc):
                mask = lb == c
                ax.scatter(emb[mask, 0], emb[mask, 1], s=6, alpha=0.6, color=cmap_pca(c))
            ax.set_title(f"{m} - {d}", fontsize=10)
            ax.set_xticks([]); ax.set_yticks([])
    plt.suptitle("PCA Features Across Layers (30 classes x 30 samples)", fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "pca_grid.png"), dpi=150, bbox_inches="tight"); plt.close()

    # Summary
    print("\n" + "-" * 65)
    print("SCENARIO 5 SUMMARY")
    print(f"{'Model':<22} {'Depth':<8} {'Val Acc':>8} {'Norm mean':>10} {'Norm std':>10}")
    for m in models:
        for d in DEPTHS:
            r = results[m][d]
            print(f"{m:<22} {d:<8} {r['acc']:>8.4f} {r['norm_m']:>10.2f} {r['norm_s']:>10.2f}")

    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_dir}")
    return results


if __name__ == "__main__":
    from config import DEFAULT_MODELS
    run(DEFAULT_MODELS)
