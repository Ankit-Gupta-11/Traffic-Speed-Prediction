import numpy as np
import pandas as pd


class Preprocessor:
    """Handle normalization and missing values."""
    
    def __init__(self):
        self.mean = None
        self.std = None
        self.is_fitted = False
    
    def fit(self, data):
        """Calculate mean and std from training data."""
        # Handle missing values first
        clean_data = self._fill_missing(data.copy())
        
        # Calculate stats
        self.mean = np.mean(clean_data, axis=0, keepdims=True)
        self.std = np.std(clean_data, axis=0, keepdims=True)
        
        # Avoid division by zero
        self.std[self.std == 0] = 1.0
        
        self.is_fitted = True
        print(f"Preprocessor fitted. Mean: {np.mean(self.mean):.2f}, Std: {np.mean(self.std):.2f}")
        
        return self
    
    def transform(self, data):
        """Normalize data using z-score."""
        if not self.is_fitted:
            raise ValueError("Call fit() before transform()")
        
        data = self._fill_missing(data.copy())
        normalized = (data - self.mean) / self.std
        return normalized.astype(np.float32)
    
    def inverse_transform(self, data):
        """Convert normalized data back to original scale."""
        if not self.is_fitted:
            raise ValueError("Call fit() before inverse_transform()")
        
        return data * self.std + self.mean
    
    def _fill_missing(self, data):
        """Fill missing values with interpolation."""
        if not np.any(np.isnan(data)):
            return data
        
        # Use pandas for easy interpolation
        df = pd.DataFrame(data)
        df = df.interpolate(method='linear', axis=0, limit_direction='both')
        df = df.fillna(method='ffill').fillna(method='bfill')
        
        # If still any NaN, fill with column mean
        if df.isna().any().any():
            df = df.fillna(df.mean())
        
        return df.values


def split_data(data, train_ratio=0.7, val_ratio=0.15):
    """Split data into train, val, test sets."""
    n = len(data)
    
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    train = data[:train_end]
    val = data[train_end:val_end]
    test = data[val_end:]
    
    print(f"Split: train={len(train)}, val={len(val)}, test={len(test)}")
    
    return train, val, test


def create_sequences(data, input_len=12, output_len=12):
    """
    Create input-output pairs for time series prediction.
    """
    num_samples = len(data) - input_len - output_len + 1
    num_features = data.shape[1]
    
    X = np.zeros((num_samples, input_len, num_features), dtype=np.float32)
    y = np.zeros((num_samples, output_len, num_features), dtype=np.float32)
    
    for i in range(num_samples):
        X[i] = data[i:i + input_len]
        y[i] = data[i + input_len:i + input_len + output_len]
    
    print(f"Created sequences: X={X.shape}, y={y.shape}")
    
    return X, y