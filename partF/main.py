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




def evaluate_model(model, loader, device, criterion, num_classes, save_path=None, softmax_path=None):
        model.eval()
        total_loss = 0.0
        all_probs = []
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for images, labels in loader:
                images, labels = images.to(device), labels.to(device)
                output = model(images)
                loss = criterion(output, labels)
                total_loss += loss.item()
                
                probs = torch.softmax(output, dim=1)
                _, preds = torch.max(output, 1)
                
                all_probs.extend(probs.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        all_probs = np.array(all_probs)
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        ap_scores = []
        acc_scores = []
        for i in range(num_classes):
            mask = (all_labels == i)
            binary_labels = mask.astype(int)
            
            ap = average_precision_score(binary_labels, all_probs[:, i])
            ap_scores.append(ap)
            
            acc = np.mean(all_preds[mask] == i)
            acc_scores.append(acc)

        avg_loss = total_loss / len(loader)
        mean_acc = 100 * np.mean(all_preds == all_labels)
        mAP = np.mean(ap_scores)

        ap_headers = ",".join([f"AP_class{i}" for i in range(num_classes)])
        acc_headers = ",".join([f"ACC_class{i}" for i in range(num_classes)])
        
        if save_path:
            with open(save_path, "w") as f:
                f.write(f"loss,total_acc,mAP,{ap_headers},{acc_headers}\n")
                ap_str = ",".join([f"{s:.4f}" for s in ap_scores])
                acc_str = ",".join([f"{s:.4f}" for s in acc_scores])
                f.write(f"{avg_loss:.4f},{mean_acc:.2f},{mAP:.4f},{ap_str},{acc_str}\n")
        
        if softmax_path:
            np.save(softmax_path, all_probs)

        return mAP, mean_acc, avg_loss, ap_scores, acc_scores, all_probs
    
    
    
    
    
    
    
def main():
    PATH_TO_DATA = "/itf-fi-ml/shared/courses/IN3310/mandatory1_data"
    PATH_TO_DATA = "../Dataset" #LOCAL SOLUTION

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
    
    model = models.resnet18(pretrained=True, weights=models.ResNet18_Weights.DEFAULT)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, NUM_CLASSES)
    model = model.to(device)
    
    

    lr, epochs, opt_name = (0.001, 25, "Adam")
    optimizer = optim.Adam(model.parameters(), lr=lr) if opt_name == "Adam" else optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"LR-{lr}_EP-{epochs}_OPT-{opt_name}_{timestamp}"
    model_name = f"{NUM_LAYERS}-ResNet-pretrained"
    save_dir = f"models/{model_name}_{run_name}"
    os.makedirs(save_dir, exist_ok=True)
    
    
    
    log_path = f"{save_dir}/log.csv"
    with open(log_path, "w") as f:
        ap_headers = ",".join([f"AP_class{i}" for i in range(NUM_CLASSES)])
        acc_headers = ",".join([f"ACC_class{i}" for i in range(NUM_CLASSES)])
        header = f"epoch,train_loss,val_loss,val_acc,mAP,{ap_headers},{acc_headers}\n"
        f.write(header)
    
    test_log_path = f"{save_dir}/test_log.csv"
    with open(test_log_path, "w") as f:
        ap_headers = ",".join([f"AP_class{i}" for i in range(NUM_CLASSES)])
        acc_headers = ",".join([f"ACC_class{i}" for i in range(NUM_CLASSES)])
        header = f"epoch,test_loss,test_acc,mAP,{ap_headers},{acc_headers}\n"
        f.write(header)
    
    
    
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
        val_mAP, val_acc, avg_val_loss, val_ap_scores, val_acc_scores, _ = evaluate_model(
            model=model,
            loader=val_loader,
            device=device,
            criterion=criterion,
            num_classes=NUM_CLASSES,
        )
                    
        with open(log_path, "a") as f:
            val_ap_str = ",".join([f"{s:.4f}" for s in val_ap_scores])
            val_acc_str = ",".join([f"{s:.4f}" for s in val_acc_scores])
            f.write(f"{epoch+1},{avg_train_loss:.4f},{avg_val_loss:.4f},{val_acc:.2f},{val_mAP:.4f},{val_ap_str},{val_acc_str}\n")
            
        test_mAP, test_acc, avg_test_loss, test_ap_scores, test_acc_scores, _ = evaluate_model(
            model=model,
            loader=test_loader,
            device=device,
            criterion=criterion,
            num_classes=NUM_CLASSES
        )
        
        with open(test_log_path, "a") as f:
            test_ap_str = ",".join([f"{s:.4f}" for s in test_ap_scores])
            test_acc_str = ",".join([f"{s:.4f}" for s in test_acc_scores])
            f.write(f"{epoch+1},{avg_test_loss:.4f},{test_acc:.2f},{test_mAP:.4f},{test_ap_str},{test_acc_str}\n")
        
    try:
        # train and loss
        df = pd.read_csv(log_path)
        df_t = pd.read_csv(test_log_path)
        
        title_desc =  f"LR-{lr}_EP-{epochs}_OPT-{opt_name}"
        
        plt.figure(figsize=(20, 15))
        
        plt.subplot(2, 2, 1)
        plt.plot(df['epoch'], df['train_loss'], label='Train Loss', color='blue')
        plt.plot(df['epoch'], df['val_loss'], label='Val Loss', color='orange')
        plt.plot(df_t['epoch'], df_t['test_loss'], label='Test Loss', color='red', linestyle='--')
        
        plt.title(f'Loss History (Train vs Val vs Test) {title_desc}')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        
        
        plt.subplot(2, 2, 2)
        plt.plot(df['epoch'], df['mAP'], label='Val mAP', color='orange')
        plt.plot(df_t['epoch'], df_t['mAP'], label='Test mAP', color='red')
        
        plt.title(f'Mean Average Precision (Val vs Test) {title_desc}')
        plt.xlabel('Epochs')
        plt.ylabel('mAP (0.0 - 1.0)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        
        
        plt.subplot(2, 2, 3)
        for i in range(NUM_CLASSES):
            plt.plot(df['epoch'], df[f'ACC_class{i}'], label=f'Class {i}', alpha=0.6, linestyle=':')
        plt.plot(df['epoch'], df['val_acc']/100, label='Mean Val Acc', color='black', lw=3)
        plt.title('Validation Accuracy Per Class', fontsize=14)
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy (0.0 - 1.0)')
        plt.legend(ncol=2, fontsize='small')
        plt.grid(True, alpha=0.3)

        plt.subplot(2, 2, 4)
        for i in range(NUM_CLASSES):
            plt.plot(df_t['epoch'], df_t[f'ACC_class{i}'], label=f'Class {i}', alpha=0.6, linestyle=':')
        plt.plot(df_t['epoch'], df_t['test_acc']/100, label='Mean Test Acc', color='black', lw=3)
        plt.title('Test Accuracy Per Class', fontsize=14)
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy (0.0 - 1.0)')
        plt.legend(ncol=2, fontsize='small')
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f"{save_dir}/training_summary_plot.png")
        plt.close()
        
    except Exception as e:
        print(f"Plotting error: {e}")
            
        

    return 0



if __name__ == "__main__":
    main()



