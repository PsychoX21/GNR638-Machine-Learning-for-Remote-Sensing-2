import argparse
import os
import re

def update_setup_bash(setup_path, old_model_id, new_model_id, new_dir_name):
    if not os.path.exists(setup_path):
        print(f"Error: {setup_path} not found.")
        return
    
    with open(setup_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Regex to find the snapshot_download model ID
    content = re.sub(
        r"snapshot_download\(\s*'[^']+',",
        f"snapshot_download(\n    '{new_model_id}',",
        content
    )
    
    # Regex to find the local_dir path
    content = re.sub(
        r"local_dir='\$\{WORK_DIR\}/weights/[^']+'",
        f"local_dir='${{WORK_DIR}}/weights/{new_dir_name}'",
        content
    )
    
    # Update echo for downloading
    content = re.sub(
        r"Downloading .* weights\.\.\.",
        f"Downloading {new_model_id} weights...",
        content
    )

    with open(setup_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Updated {setup_path} successfully.")

def update_model_manager(manager_path, new_model_id, new_dir_name, gpu_mem, max_len, enable_reasoning, enforce_eager):
    if not os.path.exists(manager_path):
        print(f"Error: {manager_path} not found.")
        return
        
    with open(manager_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Update PRIMARY_MODEL_PATH
    content = re.sub(
        r'PRIMARY_MODEL_PATH\s*=\s*"\./weights/[^"]+"',
        f'PRIMARY_MODEL_PATH = "./weights/{new_dir_name}"',
        content
    )
    
    # Update fallback HF ID
    content = re.sub(
        r'model_path\s*=\s*"[^"]+"(\s*#.*)?\n\s*logger\.warning',
        f'model_path = "{new_model_id}"\n            logger.warning',
        content
    )

    # Update logger info
    content = re.sub(
        r'Starting vLLM server for .*\.\.\.',
        f'Starting vLLM server for {new_model_id}...',
        content
    )

    # Update vLLM command arguments
    content = re.sub(
        r'"--gpu-memory-utilization",\s*"[\d\.]+"',
        f'"--gpu-memory-utilization", "{gpu_mem}"',
        content
    )
    
    content = re.sub(
        r'"--max-model-len",\s*"\d+"',
        f'"--max-model-len", "{max_len}"',
        content
    )

    # Handle reasoning flags in the cmd array
    if enable_reasoning:
        # If it's missing, we need to add it after --disable-log-stats
        if '"--reasoning-parser"' not in content:
            content = re.sub(
                r'("--disable-log-stats",?)',
                r'\1\n            "--reasoning-parser", "qwen3",',
                content
            )
    else:
        # Remove reasoning lines
        content = re.sub(r'\s*"--reasoning-parser",\s*"[^"]+",?\n?', '\n', content)

    # Handle enforce-eager flag
    if enforce_eager:
        if '"--enforce-eager"' not in content:
            content = re.sub(
                r'("--disable-log-stats",?)',
                r'\1\n            "--enforce-eager",',
                content
            )
    else:
        content = re.sub(r'\s*"--enforce-eager",?\n?', '\n', content)

    # Fix dangling commas if needed
    content = re.sub(r',\s+]', '\n        ]', content)

    with open(manager_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Updated {manager_path} successfully.")

def main():
    parser = argparse.ArgumentParser(description="Switch Pipeline Model Configuration")
    parser.add_argument("--profile", type=str, choices=["local", "prod"], default="local",
                        help="Quick configuration profile (local for 8GB VRAM, prod for L40S)")
    parser.add_argument("--model-id", type=str, help="Override HuggingFace Model ID")
    parser.add_argument("--gpu-mem", type=str, help="Override GPU memory utilization (e.g. 0.90)")
    parser.add_argument("--max-len", type=str, help="Override max model length (e.g. 4096)")
    parser.add_argument("--reasoning", action="store_true", help="Enable reasoning flags")
    parser.add_argument("--no-reasoning", action="store_true", help="Disable reasoning flags")
    
    args = parser.parse_args()

    # Defaults based on profile
    if args.profile == "local":
        model_id = "Qwen/Qwen2-VL-2B-Instruct"
        gpu_mem = "0.80"
        max_len = "6144"
        enable_reasoning = False
        enforce_eager = True
    elif args.profile == "prod":
        model_id = "Qwen/Qwen3.5-27B-FP8"
        gpu_mem = "0.88"
        max_len = "16384"
        enable_reasoning = True
        enforce_eager = False

    # Overrides
    if args.model_id:
        model_id = args.model_id
    if args.gpu_mem:
        gpu_mem = args.gpu_mem
    if args.max_len:
        max_len = args.max_len
    if args.reasoning:
        enable_reasoning = True
    elif args.no_reasoning:
        enable_reasoning = False

    new_dir_name = model_id.split("/")[-1].lower().replace("-", "_").replace(".", "")

    print(f"Applying configuration:")
    print(f"  Model ID : {model_id}")
    print(f"  Local Dir: {new_dir_name}")
    print(f"  GPU Mem  : {gpu_mem}")
    print(f"  Max Len  : {max_len}")
    print(f"  Reasoning: {'Enabled' if enable_reasoning else 'Disabled'}")
    print(f"  Eager Mode:{'Enabled' if enforce_eager else 'Disabled'}\n")

    setup_bash_path = "setup.bash"
    model_manager_path = os.path.join("src", "model_manager.py")

    update_setup_bash(setup_bash_path, None, model_id, new_dir_name)
    update_model_manager(model_manager_path, model_id, new_dir_name, gpu_mem, max_len, enable_reasoning, enforce_eager)

    print("\nDone! Please review the changes using git diff if needed.")

if __name__ == "__main__":
    main()
