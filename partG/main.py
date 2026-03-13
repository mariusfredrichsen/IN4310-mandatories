import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Subset, DataLoader 
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score, precision_score
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import datetime










def main():
    ATH_TO_DATA = "/itf-fi-ml/shared/courses/IN3310/mandatory1_data"
    # PATH_TO_DATA = "../Dataset" #LOCAL SOLUTION

    IMAGE_SIZE = 150
    IMG_CHANNELS = 3
    NUM_CLASSES = 6
    NUM_LAYERS = 18
    
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
    
    # setup data
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
    
    BATCH_SIZE = 32
    NUM_WORKERS = 0
    is_cuda = device.type == "cuda"
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=is_cuda) # pin memory loads it into the gpu, works only with cuda
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=is_cuda)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=is_cuda)
    
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, NUM_CLASSES)
    model = model.to(device)
    
    
    return 0







if __name__ == "__main__":
    main()