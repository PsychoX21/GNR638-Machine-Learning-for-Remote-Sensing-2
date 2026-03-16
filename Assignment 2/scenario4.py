# Scenario 4: Corruption Robustness Evaluation

import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim

from config import (
    seed_everything, SEED, DEVICE, RESULTS_DIR,
    LR_FINETUNE, WEIGHT_DECAY, MAX_EPOCHS_FULL, GAUSS_SIGMAS,
)
from data import create_train_val_split, get_dataloaders, corrupted_loader
from models import create_model, unfreeze_all, print_model_info
from train import train_model, evaluate

CORRUPTIONS = [
    ("clean",       "clean",       {}),
    ("gauss_005",   "gauss",       {"sigma": 0.05}),
    ("gauss_01",    "gauss",       {"sigma": 0.1}),
    ("gauss_02",    "gauss",       {"sigma": 0.2}),
    ("motion_blur", "motion_blur", {}),
    ("brightness",  "brightness",  {}),
]


def run(models: list[str], epochs: int = MAX_EPOCHS_FULL):
    """Run Scenario 4 for given models."""
    create_train_val_split()
    out_dir = os.path.join(RESULTS_DIR, "scenario4")
    os.makedirs(out_dir, exist_ok=True)
    criterion = nn.CrossEntropyLoss()
    results: dict[str, dict] = {}

    print("\n" + "=" * 70)
    print("SCENARIO 4: CORRUPTION ROBUSTNESS")
    print("=" * 70)

    for mname in models:
        print(f"\n>>> {mname} - Training on clean data ...")
        seed_everything(SEED)
        train_ld, val_ld = get_dataloaders(model_name=mname)
        model = create_model(mname).to(DEVICE)
        print_model_info(model, mname)
        unfreeze_all(model)
        opt = optim.Adam(model.parameters(), lr=LR_FINETUNE, weight_decay=WEIGHT_DECAY)
        sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        train_model(model, train_ld, val_ld, opt, sch, epochs, desc=f"S4-{mname}")

        results[mname] = {}
        for cname, ctype, ckw in CORRUPTIONS:
            if ctype == "clean":
                ld = val_ld
            else:
                ld = corrupted_loader(ctype, model_name=mname, **ckw)
            _, acc = evaluate(model, ld, criterion)
            results[mname][cname] = acc
            print(f"  {cname}: {acc:.4f}")

    # Summary table
    cnames = [c[0] for c in CORRUPTIONS]
    print("\n" + "-" * 90)
    print("SCENARIO 4 SUMMARY")
    hdr = f"{'Model':<22}" + "".join(f"{c:>14}" for c in cnames)
    print(hdr)
    for m in models:
        row = f"{m:<22}" + "".join(f"{results[m][c]:>14.4f}" for c in cnames)
        print(row)

    print(f"\n{'Model':<22} {'Corruption':<16} {'Corr Error':>12} {'Rel Robust':>12}")
    for m in models:
        cl = results[m]["clean"]
        for c in cnames[1:]:
            ce = 1 - results[m][c]
            rr = results[m][c] / cl if cl > 0 else 0
            print(f"{m:<22} {c:<16} {ce:>12.4f} {rr:>12.4f}")

    # Bar chart
    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(cnames)); w = 0.15
    for i, m in enumerate(models):
        accs = [results[m][c] for c in cnames]
        ax.bar(x + i * w, accs, w, label=m)
    ax.set_xticks(x + w * (len(models) - 1) / 2)
    ax.set_xticklabels(cnames, rotation=25, ha="right")
    ax.set_ylabel("Accuracy"); ax.set_title("Corruption Robustness")
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "robustness.png"), dpi=150); plt.close()

    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_dir}")
    return results


if __name__ == "__main__":
    from config import DEFAULT_MODELS
    run(DEFAULT_MODELS)
