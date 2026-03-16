# Scenario 2: Fine-Tuning Strategies

import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

from config import (
    seed_everything, SEED, DEVICE, RESULTS_DIR,
    LR_LINEAR_PROBE, LR_FINETUNE, WEIGHT_DECAY, MAX_EPOCHS_FULL,
)
from data import create_train_val_split, get_dataloaders
from models import (
    create_model, freeze_backbone, unfreeze_all, print_model_info,
    unfreeze_last_block, auto_unfreeze, compute_sensitivity
)
from train import train_one_epoch, evaluate, train_epoch_with_grads
from visualize import plot_grad_bars

STRATEGIES = ["linear_probe", "last_block", "full_finetune", "selective_20pct"]


def run(models: list[str], epochs: int = MAX_EPOCHS_FULL):
    """Run Scenario 2 for given models."""
    create_train_val_split()
    out_dir = os.path.join(RESULTS_DIR, "scenario2")
    os.makedirs(out_dir, exist_ok=True)
    results: dict[str, dict] = {}
    use_amp = DEVICE.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    print("\n" + "=" * 70)
    print("SCENARIO 2: FINE-TUNING STRATEGIES")
    print("=" * 70)

    for mname in models:
        results[mname] = {}
        for strat in STRATEGIES:
            print(f"\n>>> {mname} / {strat}")
            seed_everything(SEED)
            train_ld, val_ld = get_dataloaders(model_name=mname)
            model = create_model(mname).to(DEVICE)
            
            if strat == "linear_probe":
                print_model_info(model, mname)

            if strat == "linear_probe":
                freeze_backbone(model, mname)
            elif strat == "last_block":
                unfreeze_last_block(model, mname)
            elif strat == "full_finetune":
                unfreeze_all(model)
            elif strat == "selective_20pct":
                sens_scores = compute_sensitivity(model, mname, train_ld, n_batches=10)
                selected, used, total_b = auto_unfreeze(model, mname, sens_scores, max_frac=0.20)
                from visualize import plot_sensitivity
                plot_sensitivity(sens_scores, selected, mname, os.path.join(out_dir, f"{mname}_sensitivity.png"))
                print(f"  [model] Auto-selected layers: {[s[0] for s in selected]}")

            tot = sum(p.numel() for p in model.parameters())
            trn = sum(p.numel() for p in model.parameters() if p.requires_grad)
            pct = trn / tot * 100
            print(f"  Trainable: {trn:,}/{tot:,} ({pct:.1f}%)")

            lr     = LR_LINEAR_PROBE if strat == "linear_probe" else LR_FINETUNE
            params = [p for p in model.parameters() if p.requires_grad]
            opt    = optim.Adam(params, lr=lr, weight_decay=WEIGHT_DECAY)
            sch    = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
            criterion = nn.CrossEntropyLoss()
            hist   = {"tl": [], "ta": [], "vl": [], "va": []}
            gnorms_last = None

            best_acc, best_sd = 0.0, None

            import time
            t0 = time.time()
            for ep in range(1, epochs + 1):
                if ep in (1, epochs):
                    tl, ta, gn = train_epoch_with_grads(model, train_ld, criterion, opt, scaler)
                    if ep == epochs:
                        gnorms_last = gn
                else:
                    tl, ta = train_one_epoch(model, train_ld, criterion, opt, scaler)
                vl, va = evaluate(model, val_ld, criterion)
                if va > best_acc:
                    best_acc = va
                    if strat == "full_finetune":
                        best_sd = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                sch.step()
                hist["tl"].append(tl); hist["ta"].append(ta)
                hist["vl"].append(vl); hist["va"].append(va)
                if ep % 10 == 0 or ep == 1 or ep == epochs:
                    print(f"  Ep {ep:>2}/{epochs} | TrA {ta:.4f} | VaA {va:.4f}")

            elapsed = time.time() - t0
            best = best_acc # Use tracked best accuracy
            print(f"  Best Val: {best:.4f} | {elapsed/60:.1f}m")

            if strat == "full_finetune" and best_sd:
                save_path = os.path.join(out_dir, f"{mname}_full_finetune.pth")
                torch.save(best_sd, save_path)
                print(f"  [save] Saved best weights to {save_path}")

            if gnorms_last:
                plot_grad_bars(gnorms_last, mname, strat,
                               os.path.join(out_dir, f"{mname}_{strat}_grads.png"))

            results[mname][strat] = {
                "best": best, "pct": pct, "trn": trn, "time": elapsed / 60,
                "hist": hist,
            }

    # Loss convergence per model
    for mname in models:
        fig, ax = plt.subplots(figsize=(10, 6))
        for strat in STRATEGIES:
            ax.plot(range(1, epochs + 1),
                    results[mname][strat]["hist"]["tl"], label=strat)
        ax.set_xlabel("Epoch"); ax.set_ylabel("Training Loss")
        ax.set_title(f"{mname} - Loss Convergence")
        ax.legend(); ax.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{mname}_loss_conv.png"), dpi=150)
        plt.close()

    # Accuracy vs % unfrozen
    fig, ax = plt.subplots(figsize=(10, 6))
    for mname in models:
        pcts = [results[mname][s]["pct"] for s in STRATEGIES]
        accs = [results[mname][s]["best"] for s in STRATEGIES]
        ax.plot(pcts, accs, "o-", label=mname, markersize=8)
    ax.set_xlabel("% Unfrozen Parameters"); ax.set_ylabel("Best Val Accuracy")
    ax.set_title("Accuracy vs Unfrozen Parameters")
    ax.legend(); ax.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "acc_vs_unfrozen.png"), dpi=150)
    plt.close()

    # Summary
    print("\n" + "-" * 75)
    print("SCENARIO 2 SUMMARY")
    print(f"{'Model':<22} {'Strategy':<20} {'Unfrozen%':>10} {'Best Acc':>10}")
    for m in models:
        for s in STRATEGIES:
            r = results[m][s]
            print(f"{m:<22} {s:<20} {r['pct']:>10.1f} {r['best']:>10.4f}")

    # Save JSON (drop hist for brevity)
    slim = {m: {s: {k: v for k, v in r.items() if k != "hist"}
                for s, r in strats.items()} for m, strats in results.items()}
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(slim, f, indent=2)
    return results


if __name__ == "__main__":
    from config import DEFAULT_MODELS
    run(DEFAULT_MODELS)
