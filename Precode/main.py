from typing import Dict, List, Tuple
from ResNet import ResNet
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Subset, DataLoader 
import torchvision
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import datetime



def main():
    PATH_TO_DATA = "../../../../../shared/courses/IN3310/mandatory1_data"

    IMAGE_SIZE = 150
    IMG_CHANNELS = 3
    NUM_CLASSES = 6
    NUM_LAYERS = 101
    
    # A lot of this code is from the seminars
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    SEED = 42
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
    print("Seed:", SEED)
    
    data_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), # Maybe unecessary but for safe meassures
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    try:
        data_set = torchvision.datasets.ImageFolder(root=PATH_TO_DATA, transform=data_transform)
        targets = data_set.targets
    except Exception as e:
        print("Error loading data: ", e)
    
    # 
    train_idx, val_test_idx = train_test_split(
        np.arange(len(targets)), 
        test_size=0.3, 
        stratify=targets, 
        random_state=SEED)
    
    val_idx, test_idx = train_test_split(
        val_test_idx, 
        test_size=0.6, 
        stratify=[targets[i] for i in val_test_idx], 
        random_state=SEED)
    
    train_set = Subset(data_set, train_idx)
    val_set = Subset(data_set, val_idx)
    test_set = Subset(data_set, test_idx)
    
    data_size = len(data_set)
    train_size = len(train_set)
    val_size = len(val_set)
    test_size = len(test_set)
    
    print()
    print(f"Data Ready:")
    print(f"   - Training Set:   {len(train_set)} images ({(train_size / data_size) * 100:.1f}%)")
    print(f"   - Validation Set: {len(val_set)} images ({(val_size / data_size) * 100:.1f}%)")
    print(f"   - Test Set:       {len(test_set)} images ({(test_size / data_size) * 100:.1f}%)")
    print(f"   - Input Shape:    3 channels x 150 x 150 = 67500 features")
    
    print()
    print("Checking if data is setup correctly")
    
    A = set(train_set.indices)
    B = set(val_set.indices)
    C = set(test_set.indices)
    
    print(f"Overlaps train & val: {len(A.intersection(B))}")
    print(f"Overlaps train & test: {len(A.intersection(C))}")    
    print(f"Overlaps val & test: {len(B.intersection(C))}")
    
    print()
    print("Loading image data")
    
    BATCH_SIZE = 64
    NUM_WORKERS = 4
    is_cuda = device.type == "cuda"
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=is_cuda) # pin memory loads it into the gpu, works only with cuda
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=is_cuda)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=is_cuda)
    
    lr = 0.0001
    epochs = 25
    
    # setup of folder
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    model_name = f"{NUM_LAYERS}-ResNet"
    save_dir = f"models/{model_name}_{timestamp}"
    os.makedirs(save_dir, exist_ok=True)
        
    # setup of model
    model = ResNet(img_channels = IMG_CHANNELS, num_layers = NUM_LAYERS, num_classes = NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # setup of data logging
    log_path = f"{save_dir}/log.csv"
    with open(log_path, "w") as f:
        f.write("epoch,train_loss,val_loss,val_acc\n")
    
    # training
    best_acc = 0.0
    
    for epoch in range(epochs):
        print("Epoch: ", epoch)
        # part a, loading the training data
        model.train()
        train_loss = 0.0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        
        # part b, validation accuracy
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                output = model(images)
                v_loss = criterion(output, labels)
                val_loss += v_loss.item()
                
                _, predicted = torch.max(output.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            
        avg_val_loss = val_loss / len(val_loader)
        val_acc = 100 * correct / total
        
        with open(log_path, "a") as f:
            f.write(f"{epoch+1},{avg_train_loss:.4f},{avg_val_loss:.4f},{val_acc:.2f}\n")
        
        torch.save(model.state_dict(), f"{save_dir}/current_model.pth")
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), f"{save_dir}/best_model.pth")
            print(f"Epoch {epoch+1}: New best validation accuracy: {best_acc:.2f}%")
        
    def evaluate_ckpt(path):
        checkpoint = torch.load(path, weights_only=True)
        model.load_state_dict(checkpoint)
        model.eval()
        
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                output = model(images)
                _, pred = torch.max(output.data, 1)
                total += labels.size(0)
                correct += (pred == labels).sum().item()
        return 100 * correct / total

    final_test_acc = evaluate_ckpt(f"{save_dir}/best_model.pth")
    
    with open(f"{save_dir}/test_results.csv", "w") as f:
        f.write("model_version,test_acc\n")
        f.write(f"best_model,{final_test_acc:.4f}\n")
    
    try:
        df = pd.read_csv(log_path)
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.plot(df['epoch'], df['train_loss'], label='Train Loss')
        plt.plot(df['epoch'], df['val_loss'], label='Val Loss')
        plt.legend()
        plt.title('Loss History')
        
        plt.subplot(1, 2, 2)
        plt.plot(df['epoch'], df['val_acc'], color='green', marker='o')
        plt.title('Validation Accuracy')
        plt.savefig(f"{save_dir}/training_plot.png")
        # plt.show()
    except Exception as e:
        print(f"Plotting error: {e}")
    
    return 0













if __name__ == "__main__":
    main()
