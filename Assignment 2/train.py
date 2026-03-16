import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from config import DEVICE

# Core loops

def train_one_epoch(model: nn.Module, loader: DataLoader,
                    criterion: nn.Module, optimizer: optim.Optimizer,
                    scaler: torch.amp.GradScaler | None = None) -> tuple[float, float]:
    """Single training epoch with optional AMP."""
    model.train()
    loss_sum, correct, total = 0.0, 0, 0
    use_amp = scaler is not None
    for imgs, labs in loader:
        imgs, labs = imgs.to(DEVICE, non_blocking=True), labs.to(DEVICE, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=use_amp):
            out = model(imgs)
            if isinstance(out, tuple):
                out = out[0]
            loss = criterion(out, labs)
        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        loss_sum += loss.item() * imgs.size(0)
        correct  += out.argmax(1).eq(labs).sum().item()
        total    += labs.size(0)
    return loss_sum / total, correct / total


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader,
             criterion: nn.Module) -> tuple[float, float]:
    """Evaluate model; returns (avg_loss, accuracy)."""
    model.eval()
    loss_sum, correct, total = 0.0, 0, 0
    for imgs, labs in loader:
        imgs, labs = imgs.to(DEVICE, non_blocking=True), labs.to(DEVICE, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=DEVICE.type == "cuda"):
            out = model(imgs)
            if isinstance(out, tuple):
                out = out[0]
            loss = criterion(out, labs)
        loss_sum += loss.item() * imgs.size(0)
        correct  += out.argmax(1).eq(labs).sum().item()
        total    += labs.size(0)
    return loss_sum / total, correct / total

# Full training

def train_model(model: nn.Module, train_ld: DataLoader, val_ld: DataLoader,
                optimizer: optim.Optimizer, scheduler, epochs: int,
                desc: str = "", use_amp: bool = True) -> tuple[dict, float, float]:
    """Train for `epochs`, track best val accuracy, return (history, elapsed_s, best_acc)."""
    criterion = nn.CrossEntropyLoss()
    hist: dict[str, list[float]] = {"tl": [], "ta": [], "vl": [], "va": []}
    best_acc, best_sd = 0.0, None
    scaler = torch.amp.GradScaler("cuda") if (use_amp and DEVICE.type == "cuda") else None
    t0 = time.time()

    for ep in range(1, epochs + 1):
        tl, ta = train_one_epoch(model, train_ld, criterion, optimizer, scaler)
        vl, va = evaluate(model, val_ld, criterion)
        if scheduler:
            scheduler.step()
        hist["tl"].append(tl); hist["ta"].append(ta)
        hist["vl"].append(vl); hist["va"].append(va)
        if va > best_acc:
            best_acc = va
            best_sd  = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if ep % 5 == 0 or ep == 1 or ep == epochs:
            print(f"  [{desc}] Ep {ep:>2}/{epochs} | TrL {tl:.4f} TrA {ta:.4f} | VaL {vl:.4f} VaA {va:.4f}")

    elapsed = time.time() - t0
    print(f"  [{desc}] Done {elapsed/60:.1f}m | Best Val {best_acc:.4f}")
    if best_sd:
        model.load_state_dict(best_sd)
    return hist, elapsed, best_acc

# Gradient tracking

def compute_grad_norms(model: nn.Module) -> dict[str, float]:
    """Return L2 gradient norm per named parameter."""
    return {
        n: p.grad.data.norm(2).item()
        for n, p in model.named_parameters()
        if p.grad is not None
    }


def train_epoch_with_grads(model: nn.Module, loader: DataLoader,
                           criterion: nn.Module, optimizer: optim.Optimizer,
                           scaler: torch.amp.GradScaler | None = None):
    """Train one epoch and collect batch-averaged gradient norms."""
    model.train()
    loss_sum, correct, total = 0.0, 0, 0
    all_norms: dict[str, list[float]] = {}
    use_amp = scaler is not None

    for imgs, labs in loader:
        imgs, labs = imgs.to(DEVICE, non_blocking=True), labs.to(DEVICE, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=use_amp):
            out = model(imgs)
            if isinstance(out, tuple):
                out = out[0]
            loss = criterion(out, labs)
        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        gn = compute_grad_norms(model)
        for k, v in gn.items():
            all_norms.setdefault(k, []).append(v)
        loss_sum += loss.item() * imgs.size(0)
        correct  += out.argmax(1).eq(labs).sum().item()
        total    += labs.size(0)

    avg = {k: float(np.mean(v)) for k, v in all_norms.items()}
    return loss_sum / total, correct / total, avg

# Feature extraction

@torch.no_grad()
def extract_features(model: nn.Module, loader: DataLoader) -> tuple[np.ndarray, np.ndarray]:
    """Extract penultimate-layer features via timm's forward_features (GAP'd)."""
    model.eval()
    feats_list, labs_list = [], []
    for imgs, labels in loader:
        imgs = imgs.to(DEVICE, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=DEVICE.type == "cuda"):
            f = model.forward_features(imgs)
        if f.dim() == 4:
            f = f.mean([2, 3])
        elif f.dim() == 3:
            f = f.mean(1)
        feats_list.append(f.float().cpu().numpy())
        labs_list.append(labels.numpy())
    return np.concatenate(feats_list), np.concatenate(labs_list)


@torch.no_grad()
def extract_at_layer(model: nn.Module, layer_module: nn.Module,
                     loader: DataLoader) -> tuple[np.ndarray, np.ndarray]:
    """Extract features from a specific layer using a forward hook."""
    model.eval()
    activation: dict[str, torch.Tensor] = {}

    def hook(m, inp, out):
        activation["f"] = out

    handle = layer_module.register_forward_hook(hook)
    feats_list, labs_list = [], []
    for imgs, labels in loader:
        imgs = imgs.to(DEVICE, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=DEVICE.type == "cuda"):
            _ = model(imgs)
        f = activation["f"]
        if isinstance(f, tuple):
            f = f[0]
        if f.dim() == 4:
            f = f.mean([2, 3])
        elif f.dim() == 3:
            f = f.mean(1)
        feats_list.append(f.float().cpu().numpy())
        labs_list.append(labels.numpy())
    handle.remove()
    return np.concatenate(feats_list), np.concatenate(labs_list)
