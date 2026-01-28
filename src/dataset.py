import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class TrafficDataset(Dataset):
    """Simple PyTorch Dataset for traffic prediction."""
    
    def __init__(self, X, y):

        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def get_dataloaders(X_train, y_train, X_val, y_val, X_test, y_test, batch_size=64):
    """Create train, val, test dataloaders."""
    
    train_dataset = TrafficDataset(X_train, y_train)
    val_dataset = TrafficDataset(X_val, y_val)
    test_dataset = TrafficDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"DataLoaders created: train={len(train_loader)} batches, val={len(val_loader)}, test={len(test_loader)}")
    
    return train_loader, val_loader, test_loader