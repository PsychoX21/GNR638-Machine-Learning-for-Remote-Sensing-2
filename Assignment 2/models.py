# Model factory: creation, freezing/unfreezing, probes, and efficiency metrics.

import timm
import torch
import torch.nn as nn

from config import NUM_CLASSES, IMG_SIZE, INCEPTION_SIZE

try:
    from ptflops import get_model_complexity_info
    HAS_PTFLOPS = True
except ImportError:
    HAS_PTFLOPS = False

# Model creation

def create_model(name: str, pretrained: bool = True) -> nn.Module:
    """Create a timm model with a new 30-class head."""
    return timm.create_model(name, pretrained=pretrained, num_classes=NUM_CLASSES)


def _format_count(n: float) -> str:
    """Format a large number with SI suffix."""
    for suffix in ["", "K", "M", "G", "T"]:
        if abs(n) < 1000:
            return f"{n:.2f} {suffix}" if suffix else f"{n:.0f}"
        n /= 1000
    return f"{n:.2f} P"


def print_model_info(model: nn.Module, name: str):
    """Print parameter counts, MACs, and FLOPs (= 2 x MACs)."""
    total = sum(p.numel() for p in model.parameters())
    train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    size  = INCEPTION_SIZE if name == "inception_v3" else IMG_SIZE
    print(f"\n{'='*55}")
    print(f"  {name}")
    print(f"  Input size      : {size}x{size}")
    print(f"  Total params    : {total:>12,}")
    print(f"  Trainable params: {train:>12,}")
    print(f"  Frozen params   : {total - train:>12,}")
    if HAS_PTFLOPS:
        macs_str, _ = get_model_complexity_info(
            model, (3, size, size),
            as_strings=True, print_per_layer_stat=False, verbose=False,
        )
        macs_num, _ = get_model_complexity_info(
            model, (3, size, size),
            as_strings=False, print_per_layer_stat=False, verbose=False,
        )
        flops_num = macs_num * 2
        print(f"  MACs            : {macs_str}")
        print(f"  FLOPs (approx)  : {_format_count(flops_num)}FLOPs")
    print(f"{'='*55}")

# Classifier keys

_CLS_KEYS = {
    "resnet50":        ["fc."],
    "inception_v3":    ["fc."],
    "densenet121":     ["classifier."],
    "efficientnet_b0": ["classifier."],
    "convnext_tiny":   ["head.fc.", "head.norm."],
}

def _is_classifier(param_name: str, model_name: str) -> bool:
    return any(k in param_name for k in _CLS_KEYS.get(model_name, ["fc.", "classifier.", "head."]))

# Freeze / Unfreeze

def freeze_backbone(model: nn.Module, model_name: str):
    """Freeze all parameters except the classifier head."""
    for n, p in model.named_parameters():
        if not _is_classifier(n, model_name):
            p.requires_grad = False


def unfreeze_all(model: nn.Module):
    """Unfreeze every parameter."""
    for p in model.parameters():
        p.requires_grad = True

# Layer groups

def get_layer_groups(model: nn.Module, model_name: str) -> list[tuple[str, list[nn.Parameter]]]:
    """Return named groups of backbone parameters for fine-tuning control."""
    groups = []
    if model_name == "resnet50":
        groups.append(("stem", [p for n, p in model.named_parameters()
                                if n.startswith(("conv1", "bn1"))]))
        for i in range(1, 5):
            groups.append((f"layer{i}", [p for n, p in model.named_parameters()
                                         if n.startswith(f"layer{i}")]))
    elif model_name == "inception_v3":
        blocks: dict[str, list] = {}
        for n, p in model.named_parameters():
            if _is_classifier(n, model_name):
                continue
            top = n.split('.')[0]
            blocks.setdefault(top, []).append(p)
        for k, v in blocks.items():
            groups.append((k, v))
    elif model_name == "densenet121":
        groups.append(("stem", [p for n, p in model.named_parameters()
                                if n.startswith("features.conv0") or n.startswith("features.norm0")]))
        for i in range(1, 5):
            params = [p for n, p in model.named_parameters()
                      if n.startswith(f"features.denseblock{i}") or n.startswith(f"features.transition{i}")]
            if params:
                groups.append((f"denseblock{i}", params))
        norm5 = [p for n, p in model.named_parameters() if n.startswith("features.norm5")]
        if norm5:
            groups.append(("norm5", norm5))
    elif model_name == "efficientnet_b0":
        groups.append(("stem", [p for n, p in model.named_parameters()
                                if n.startswith("conv_stem") or n.startswith("bn1")]))
        blocks_dict: dict[str, list] = {}
        for n, p in model.named_parameters():
            if n.startswith("blocks."):
                idx = n.split('.')[1]
                blocks_dict.setdefault(f"block{idx}", []).append(p)
        for k in sorted(blocks_dict):
            groups.append((k, blocks_dict[k]))
        tail = [p for n, p in model.named_parameters()
                if n.startswith("conv_head") or n.startswith("bn2")]
        if tail:
            groups.append(("head_conv", tail))
    elif model_name == "convnext_tiny":
        groups.append(("stem", [p for n, p in model.named_parameters() if n.startswith("stem")]))
        for i in range(4):
            params = [p for n, p in model.named_parameters() if n.startswith(f"stages.{i}")]
            if params:
                groups.append((f"stage{i}", params))
    return groups


def unfreeze_last_block(model: nn.Module, model_name: str):
    """Freeze backbone, then unfreeze only the last layer group + classifier."""
    freeze_backbone(model, model_name)
    groups = get_layer_groups(model, model_name)
    if groups:
        name, params = groups[-1]
        for p in params:
            p.requires_grad = True
        print(f"  [model] Unfroze last block: {name} ({sum(p.numel() for p in params):,} params)")


def unfreeze_selective(model: nn.Module, model_name: str, max_frac: float = 0.20):
    """Unfreeze up to max_frac of backbone params, deepest layers first."""
    freeze_backbone(model, model_name)
    groups = get_layer_groups(model, model_name)
    total  = sum(sum(p.numel() for p in ps) for _, ps in groups)
    budget = int(total * max_frac)
    used, selected = 0, []
    for gname, params in reversed(groups):
        sz = sum(p.numel() for p in params)
        if used + sz <= budget:
            for p in params:
                p.requires_grad = True
            used += sz
            selected.append(gname)
    print(f"  [model] Selective unfreeze: {used:,}/{total:,} ({used/total*100:.1f}%) -> {selected}")
    return selected

# Probe layers (Scenario 5)

def get_probe_layers(model: nn.Module, model_name: str) -> dict[str, nn.Module]:
    """Return early/middle/final layer modules for feature extraction hooks."""
    if model_name == "resnet50":
        return {"early": model.layer1, "middle": model.layer3, "final": model.layer4}
    elif model_name == "inception_v3":
        return {"early": model.Mixed_5b, "middle": model.Mixed_6a, "final": model.Mixed_7c}
    elif model_name == "densenet121":
        return {
            "early":  model.features.denseblock1,
            "middle": model.features.denseblock3,
            "final":  model.features.denseblock4,
        }
    elif model_name == "efficientnet_b0":
        return {"early": model.blocks[1], "middle": model.blocks[4], "final": model.blocks[6]}
    elif model_name == "convnext_tiny":
        return {"early": model.stages[0], "middle": model.stages[2], "final": model.stages[3]}
    raise ValueError(f"Unknown model: {model_name}")

# Sensitivity (Bonus)

def compute_sensitivity(model: nn.Module, model_name: str, loader, n_batches: int = 20):
    """Compute normalized gradient magnitude per layer group as a sensitivity score."""
    import numpy as np
    model.train()
    for p in model.parameters():
        p.requires_grad = True
    criterion = nn.CrossEntropyLoss()
    groups = get_layer_groups(model, model_name)
    sens: dict[str, list[float]] = {n: [] for n, _ in groups}
    device = next(model.parameters()).device
    for i, (imgs, labs) in enumerate(loader):
        if i >= n_batches:
            break
        imgs, labs = imgs.to(device), labs.to(device)
        model.zero_grad()
        out = model(imgs)
        if isinstance(out, tuple):
            out = out[0]
        criterion(out, labs).backward()
        for gn, ps in groups:
            gn_val = sum(p.grad.data.norm(2).item() for p in ps if p.grad is not None)
            np_val = sum(p.numel() for p in ps)
            sens[gn].append(gn_val / (np_val ** 0.5 + 1e-8))
    return {k: float(np.mean(v)) for k, v in sens.items() if v}


def auto_unfreeze(model: nn.Module, model_name: str,
                  sens_scores: dict[str, float], max_frac: float = 0.20):
    """Greedily unfreeze highest-sensitivity layers within the parameter budget."""
    for p in model.parameters():
        p.requires_grad = False
    for n, p in model.named_parameters():
        if _is_classifier(n, model_name):
            p.requires_grad = True
    groups = get_layer_groups(model, model_name)
    total  = sum(sum(p.numel() for p in ps) for _, ps in groups)
    budget = int(total * max_frac)
    ranked = sorted(
        [(n, ps, sens_scores.get(n, 0)) for n, ps in groups],
        key=lambda x: x[2], reverse=True,
    )
    used, selected = 0, []
    for gn, ps, sc in ranked:
        sz = sum(p.numel() for p in ps)
        if used + sz <= budget:
            for p in ps:
                p.requires_grad = True
            used += sz
            selected.append((gn, sc))
    return selected, used, total
