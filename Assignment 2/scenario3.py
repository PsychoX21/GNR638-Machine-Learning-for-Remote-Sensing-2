# Scenario 3: Few-Shot Learning Analysis

import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch.optim as optim

from config import (
    seed_everything, SEED, DEVICE, RESULTS_DIR,
    LR_FINETUNE, WEIGHT_DECAY, MAX_EPOCHS_FULL, MAX_EPOCHS_FEW,
    FEWSHOT_FRACS,
)
from data import create_train_val_split, get_dataloaders
from models import create_model, unfreeze_all, print_model_info
from train import train_model


def run(models: list[str], epochs_full: int = MAX_EPOCHS_FULL,
        epochs_few: int = MAX_EPOCHS_FEW):
    """Run Scenario 3 for given models."""
    create_train_val_split()
    out_dir = os.path.join(RESULTS_DIR, "scenario3")
    os.makedirs(out_dir, exist_ok=True)
    results: dict[str, dict] = {}

    print("\n" + "=" * 70)
    print("SCENARIO 3: FEW-SHOT LEARNING ANALYSIS")
    print("=" * 70)

    for mname in models:
        results[mname] = {}
        for frac in FEWSHOT_FRACS:
            n_ep = epochs_full if frac == 1.0 else epochs_few
            print(f"\n>>> {mname} / {frac*100:.0f}% data / {n_ep} epochs")
            seed_everything(SEED)
            train_ld, val_ld = get_dataloaders(model_name=mname, fraction=frac)
            model = create_model(mname).to(DEVICE)
            if frac == 1.0:
                print_model_info(model, mname)
            unfreeze_all(model)
            opt = optim.Adam(model.parameters(), lr=LR_FINETUNE, weight_decay=WEIGHT_DECAY)
            sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_ep)
            hist, elapsed, best = train_model(
                model, train_ld, val_ld, opt, sch, n_ep,
                desc=f"FS-{mname}-{frac*100:.0f}%",
            )
            results[mname][str(frac)] = {
                "best":        best,
                "time":        elapsed / 60,
                "final_train": hist["ta"][-1],
                "gap":         hist["ta"][-1] - hist["va"][-1],
            }

    # Summary
    print("\n" + "-" * 80)
    print("SCENARIO 3 SUMMARY")
    print(f"{'Model':<22} {'100%':>8} {'20%':>8} {'5%':>8} {'Delta':>8} {'Gap@5%':>8}")
    for m in models:
        a1    = results[m]["1.0"]["best"]
        a2    = results[m]["0.2"]["best"]
        a3    = results[m]["0.05"]["best"]
        delta = (a1 - a3) / a1 if a1 > 0 else 0
        gap   = results[m]["0.05"]["gap"]
        print(f"{m:<22} {a1:>8.4f} {a2:>8.4f} {a3:>8.4f} {delta:>8.4f} {gap:>8.4f}")

    # Accuracy vs % data
    fig, ax = plt.subplots(figsize=(10, 6))
    for m in models:
        ax.plot([f * 100 for f in FEWSHOT_FRACS],
                [results[m][str(f)]["best"] for f in FEWSHOT_FRACS],
                "o-", label=m, ms=8)
    ax.set_xlabel("% Training Data"); ax.set_ylabel("Best Val Accuracy")
    ax.set_title("Few-Shot Performance"); ax.legend(); ax.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fewshot.png"), dpi=150); plt.close()

    # Overfitting gap
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(models)); w = 0.25
    for i, frac in enumerate(FEWSHOT_FRACS):
        gaps = [results[m][str(frac)]["gap"] for m in models]
        ax.bar(x + i * w, gaps, w, label=f"{frac*100:.0f}%")
    ax.set_xticks(x + w); ax.set_xticklabels(models, rotation=30, ha="right")
    ax.set_ylabel("Train - Val Gap")
    ax.set_title("Overfitting Analysis"); ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "gap.png"), dpi=150); plt.close()

    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_dir}")
    return results


if __name__ == "__main__":
    from config import DEFAULT_MODELS
    run(DEFAULT_MODELS)
