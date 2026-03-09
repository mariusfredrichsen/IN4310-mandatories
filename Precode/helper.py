import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os

def main():
    # 1. Setup the argument parser
    parser = argparse.ArgumentParser(description='Plot training logs from a CSV file.')
    parser.add_argument('file', type=str, help='Path to the log.csv file')
    parser.add_argument('--title', type=str, default='Model Training', help='Title for the plots')
    
    args = parser.parse_args()

    # 2. Load the data
    if not os.path.exists(args.file):
        print(f"Error: File '{args.file}' not found.")
        return

    df = pd.read_csv(args.file)

    # 3. Create the figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(args.title, fontsize=16)

    # --- Plot 1: Loss ---
    ax1.plot(df['epoch'], df['train_loss'], label='Train Loss', color='#1f77b4', linewidth=2)
    ax1.plot(df['epoch'], df['val_loss'], label='Val Loss', color='#ff7f0e', linestyle='--', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss History')
    ax1.legend()
    ax1.grid(True, linestyle=':', alpha=0.6)

    # --- Plot 2: Accuracy ---
    ax2.plot(df['epoch'], df['val_acc'], label='Val Accuracy', color='#2ca02c', marker='o', markersize=4)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Validation Accuracy')
    ax2.legend()
    ax2.grid(True, linestyle=':', alpha=0.6)

    # 4. Save the plot
    # This automatically names the image based on the input filename
    output_path = args.file.replace('.csv', '_plot.png')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust for the suptitle
    plt.savefig(output_path, dpi=300)
    
    print(f"✅ Success! Plot saved to: {output_path}")

if __name__ == "__main__":
    main()
