import argparse
import sys
import os

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(__file__))


def parse_args():
    p = argparse.ArgumentParser(
        description="GNR638 Assignment 2 - Pre-trained CNN Transfer & Robustness",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--scenario", nargs="+", default=["all"],
        help="Scenario(s) to run: 1 2 3 4 5 or 'all' (default: all)",
    )
    p.add_argument(
        "--models", nargs="+", default=None,
        help="Model name(s): resnet50 inception_v3 densenet121 efficientnet_b0 convnext_tiny or 'all'",
    )
    p.add_argument("--epochs", type=int, default=None, help="Override max epochs for full-data training")
    p.add_argument("--epochs-few", type=int, default=None, help="Override max epochs for few-shot training")
    p.add_argument("--data-dir", type=str, default=None, help="Path to dataset root (image-folder format)")
    p.add_argument("--results-dir", type=str, default=None, help="Path to save results")
    p.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    p.add_argument("--seed", type=int, default=None, help="Override random seed")
    p.add_argument("--setup", action="store_true", help="Only create train/val split, then exit")
    return p.parse_args()


def main():
    args = parse_args()

    # Apply overrides BEFORE importing config-dependent modules
    if args.data_dir:
        os.environ["DATA_DIR"] = args.data_dir
    if args.results_dir:
        os.environ["RESULTS_DIR"] = args.results_dir
    if args.batch_size:
        os.environ["BATCH_SIZE"] = str(args.batch_size)
    if args.seed:
        os.environ["SEED"] = str(args.seed)

    # Now import (config reads env vars at import time)
    import config
    from data import create_train_val_split

    config.seed_everything(config.SEED)
    print(f"Device : {config.DEVICE}")
    if config.DEVICE.type == "cuda":
        import torch
        print(f"GPU    : {torch.cuda.get_device_name(0)}")
        print(f"VRAM   : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"Seed   : {config.SEED}")
    print(f"Data   : {config.DATA_DIR}")
    print(f"Results: {config.RESULTS_DIR}")
    os.makedirs(config.RESULTS_DIR, exist_ok=True)

    # Setup only
    if args.setup:
        create_train_val_split()
        print("Setup complete.")
        return

    # Resolve models
    if args.models is None:
        models = config.DEFAULT_MODELS
    elif "all" in args.models:
        models = config.ALL_MODELS
    else:
        models = args.models
    print(f"Models : {models}")

    # Resolve epochs
    epochs_full = args.epochs if args.epochs else config.MAX_EPOCHS_FULL
    epochs_few  = args.epochs_few if args.epochs_few else config.MAX_EPOCHS_FEW

    # Resolve scenarios
    if "all" in args.scenario:
        scenarios = [1, 2, 3, 4, 5]
    else:
        scenarios = [int(s) for s in args.scenario]
    print(f"Scenarios: {scenarios}")
    print()

    # Run
    if 1 in scenarios:
        import scenario1
        scenario1.run(models, epochs=epochs_full)

    if 2 in scenarios:
        import scenario2
        scenario2.run(models, epochs=epochs_full)

    if 3 in scenarios:
        import scenario3
        scenario3.run(models, epochs_full=epochs_full, epochs_few=epochs_few)

    if 4 in scenarios:
        import scenario4
        scenario4.run(models, epochs=epochs_full)

    if 5 in scenarios:
        import scenario5
        scenario5.run(models)

    print("\n" + "=" * 70)
    print("ALL DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
