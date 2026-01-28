import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import time


class Trainer:
    """Simple trainer for LSTM models."""
    
    def __init__(self, model, device='cpu', lr=0.001, weight_decay=1e-4):
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.MSELoss()
        # self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        
        # For tracking history
        self.train_losses = []
        self.val_losses = []
    
    def train_epoch(self, train_loader):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            predictions = self.model(X_batch)
            loss = self.criterion(predictions, y_batch)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(train_loader)
    
    def validate(self, val_loader):
        """Validate the model."""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                predictions = self.model(X_batch)
                loss = self.criterion(predictions, y_batch)
                total_loss += loss.item()
        
        return total_loss / len(val_loader)
    
    def train(self, train_loader, val_loader, epochs=50, patience=10):
        """
        Full training loop with early stopping.
        """
        print(f"Training on {self.device}...")
        print(f"Epochs: {epochs}, Patience: {patience}")
        print("-" * 50)
        
        best_val_loss = float('inf')
        best_model_state = None
        patience_counter = 0
        
        start_time = time.time()
        
        for epoch in range(1, epochs + 1):
            # Train
            train_loss = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)
            
            # Validate
            val_loss = self.validate(val_loader)
            self.val_losses.append(val_loss)
            
            # Print progress
            print(f"Epoch {epoch:3d}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            
            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = self.model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"\nEarly stopping at epoch {epoch}")
                    break
        
        # Restore best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
        
        total_time = time.time() - start_time
        print("-" * 50)
        print(f"Training completed in {total_time/60:.1f} minutes")
        print(f"Best validation loss: {best_val_loss:.4f}")
        
        return self.train_losses, self.val_losses
    
    def save_model(self, path):
        """Save model weights."""
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")
    
    def load_model(self, path):
        """Load model weights."""
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        print(f"Model loaded from {path}")