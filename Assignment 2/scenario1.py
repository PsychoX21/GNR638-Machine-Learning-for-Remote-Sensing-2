# Scenario 1: Linear Probe Transfer

import os, json
import torch.optim as optim
from config import (
    seed_everything, SEED, DEVICE, RESULTS_DIR,
    LR_LINEAR_PROBE, WEIGHT_DECAY, MAX_EPOCHS_FULL,
)
from data import create_train_val_split, get_dataloaders, get_class_names
from models import create_model, freeze_backbone, print_model_info
from train import train_model, extract_features
from visualize import plot_curves, plot_confusion_matrix, plot_embeddings


def run(models: list[str], epochs: int = MAX_EPOCHS_FULL):
    """Run Scenario 1 for given models."""
    create_train_val_split()
    class_names = get_class_names()
    out_dir = os.path.join(RESULTS_DIR, "scenario1")
    os.makedirs(out_dir, exist_ok=True)
    results = {}

    print("\n" + "=" * 70)
    print("SCENARIO 1: LINEAR PROBE TRANSFER")
    print("=" * 70)

    for mname in models:
        print(f"\n>>> {mname}")
        seed_everything(SEED)
        train_ld, val_ld = get_dataloaders(model_name=mname, fraction=1.0)
        model = create_model(mname).to(DEVICE)
        freeze_backbone(model, mname)
        print_model_info(model, mname)

        params = [p for p in model.parameters() if p.requires_grad]
        opt = optim.Adam(params, lr=LR_LINEAR_PROBE, weight_decay=WEIGHT_DECAY)
        sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

        hist, elapsed, best = train_model(
            model, train_ld, val_ld, opt, sch, epochs, desc=f"LP-{mname}",
        )

        # plots
        plot_curves(hist, f"Linear Probe - {mname}",
                    os.path.join(out_dir, f"{mname}_curves.png"))
        plot_confusion_matrix(model, val_ld, class_names,
                              f"Linear Probe - {mname}",
                              os.path.join(out_dir, f"{mname}_cm.png"))
        feats, labs = extract_features(model, val_ld)
        plot_embeddings(feats, labs, class_names, mname, out_dir)

        results[mname] = {"best_val": best, "time_min": elapsed / 60}

    # summary
    print("\n" + "-" * 55)
    print("SCENARIO 1 SUMMARY")
    print(f"{'Model':<22} {'Best Val Acc':>12} {'Time (min)':>10}")
    for m, r in results.items():
        print(f"{m:<22} {r['best_val']:>12.4f} {r['time_min']:>10.1f}")

    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_dir}")
    return results


if __name__ == "__main__":
    from config import DEFAULT_MODELS
    run(DEFAULT_MODELS)
