import re
import matplotlib.pyplot as plt
import argparse
import os

# Increase font size globally
plt.rcParams.update({'font.size': 14})

def parse_logs(log_content):
    """
    Parses logs in the format:
    Epoch [1] Batch [0/352] Loss: 6.1308 Acc: 1.56%
    ...
    Epoch 1/50 | Train Loss: 2.3456 | Train Acc: 12.34% | Time: 12.56s
                 | Val Loss: 2.1234 | Val Acc: 15.67%
    """
    epochs = []
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    
    # Regex for epoch validation summary
    # Match: Epoch 1/40 (across multiple lines potentially)
    # Then: Train Loss: 0.2126, Train Acc: 93.44%
    # Then: Val Loss: 0.1187, Val Acc: 96.53%
    
    # We will iterate line by line and look for the specific summary block
    # ======================================================================
    # Epoch 1/40
    # Train Loss: 0.2126, Train Acc: 93.44%
    # Val Loss: 0.1187, Val Acc: 96.53%
    # Time: 108.50s
    # ======================================================================

    summary_pattern_header = re.compile(r"Epoch (\d+)/(\d+)")
    train_pattern = re.compile(r"Train Loss:\s+([\d\.]+),\s+Train Acc:\s+([\d\.]+)%")
    val_pattern = re.compile(r"Val Loss:\s+([\d\.]+),\s+Val Acc:\s+([\d\.]+)%")

    lines = log_content.split('\n')
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        # Check for Epoch header in summary block
        header_match = summary_pattern_header.match(line)
        if header_match:
            # We found "Epoch X/Y"
            current_epoch = int(header_match.group(1))
            
            # Look ahead for Train/Val stats
            # Usually next line or next few lines
            found_train = False
            found_val = False
            
            # Scan next few lines
            for j in range(1, 4):
                if i + j >= len(lines): break
                next_line = lines[i+j].strip()
                
                t_match = train_pattern.search(next_line)
                if t_match:
                    epochs.append(current_epoch)
                    train_losses.append(float(t_match.group(1)))
                    train_accs.append(float(t_match.group(2)))
                    found_train = True
                
                v_match = val_pattern.search(next_line)
                if v_match:
                    val_losses.append(float(v_match.group(1)))
                    val_accs.append(float(v_match.group(2)))
                    found_val = True
            
            if found_train:
                # Advance i to skip these lines
                i += 1
                continue

        i += 1

    return epochs, train_losses, train_accs, val_losses, val_accs

def plot_logs(epochs, train, val, title, ylabel, filename):
    plt.figure(figsize=(12, 8))
    plt.plot(epochs, train, label=f'Train {ylabel}', linewidth=3)
    if val and len(val) == len(epochs):
        plt.plot(epochs, val, label=f'Val {ylabel}', linewidth=3, linestyle='--')
    
    plt.title(title, fontsize=18)
    plt.xlabel('Epochs', fontsize=16)
    plt.ylabel(ylabel, fontsize=16)
    plt.legend(fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    print(f"Saved {filename}")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Parse training logs and generate plots")
    parser.add_argument('logfile', type=str, help="Path to text file containing training terminal output")
    parser.add_argument('--name', type=str, default="Model", help="Name of the model/dataset for titles")
    args = parser.parse_args()
    
    if not os.path.exists(args.logfile):
        print(f"Error: File {args.logfile} not found.")
        return

    with open(args.logfile, 'r') as f:
        content = f.read()
        
    epochs, t_loss, t_acc, v_loss, v_acc = parse_logs(content)
    
    if not epochs:
        print("No epoch data found in logs. Make sure to paste the full output starting from 'Epoch 1/...'")
        return
        
    print(f"Found {len(epochs)} epochs.")
    
    os.makedirs('report/images', exist_ok=True)
    
    # Plot Loss
    plot_logs(epochs, t_loss, v_loss, 
              f'{args.name} Training Loss', 'Loss', 
              f'report/images/{args.name}_real_loss.png')
              
    # Plot Accuracy
    plot_logs(epochs, t_acc, v_acc, 
              f'{args.name} Training Accuracy', 'Accuracy (%)', 
              f'report/images/{args.name}_real_acc.png')

if __name__ == "__main__":
    main()
