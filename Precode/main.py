from typing import Dict, List, Tuple
from ResNet import ResNet
import os
import torch
import torchvision
import torchvision.transforms as transforms


def load_data(path_to_data: str) -> List[List[str]]:
    return [os.listdir(os.path.join(path_to_data,class_folder)) for class_folder in os.listdir(path_to_data)]

def main():
    PATH_TO_DATA = "../Dataset"
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    
    all_data = load_data(PATH_TO_DATA)
    classes = ('buildings', 'forest', 'glacier', 'mountain', 'sea', 'street')
    
    IMAGE_SIZE = 150
    IMG_CHANNELS = 3
    NUM_CLASSES = len(classes)
    
    # A lot of this code is from the seminars
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    SEED = 42
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
    print("Seed:", SEED)
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    try:
        full_train_set = torchvision(root='../Dataset', train=True, download=True, transform=transform)
        test_set = torchvision(root='../Dataset', train=False, download=True, transform=transform)
    except:
        print("Error loading data")
    
    train_size = int(0.7 * len(full_train_set))
    val_size = len(full_train_set) - train_size
    train_set, val_set = torch.utils.data.random_split(full_train_set, [train_size, val_size])
    
    BATCH_SIZE = 128
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)
        
    print()
    print(f"Data Ready:")
    print(f"   - Training Set:   {len(train_set)} images")
    print(f"   - Validation Set: {len(val_set)} images")
    print(f"   - Test Set:       {len(test_set)} images")
    print(f"   - Input Shape:    3 channels x 150 x 150 = 67500 features")
    
    print()
    print("Checking if data is setup correctly")
        
    model = ResNet(img_channels = IMG_CHANNELS, num_layers = 18, num_classes = NUM_CLASSES)
    
    return 0













if __name__ == "__main__":
    main()