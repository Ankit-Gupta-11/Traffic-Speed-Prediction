import numpy as np
import pandas as pd
import h5py
import os


class DataLoader:
    """Load METR-LA traffic speed data from HDF5 file."""
    
    def __init__(self, data_path):
        self.data_path = data_path
        self.data = None
        self.timestamps = None
        self.sensor_ids = None
    
    def load(self):
        """Load data from HDF5 file."""
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
        
        print(f"Loading data from {self.data_path}...")
        
        with h5py.File(self.data_path, 'r') as f:
            # Load speed data
            self.data = np.array(f['df/block0_values'], dtype=np.float32)
            
            # Load timestamps
            timestamps_raw = np.array(f['df/axis1'])
            self.timestamps = pd.to_datetime(timestamps_raw)
            
            # Load sensor IDs
            self.sensor_ids = np.array(f['df/axis0']).astype(str)
        
        print(f"Loaded: {self.data.shape[0]} timesteps, {self.data.shape[1]} sensors")
        print(f"Time range: {self.timestamps[0]} to {self.timestamps[-1]}")
        
        return self.data, self.timestamps
    
    def get_info(self):
        """Print basic info about the dataset."""
        if self.data is None:
            print("Data not loaded yet. Call load() first.")
            return
        
        print("\n--- Dataset Info ---")
        print(f"Shape: {self.data.shape}")
        print(f"Mean speed: {np.nanmean(self.data):.2f} mph")
        print(f"Std speed: {np.nanstd(self.data):.2f} mph")
        print(f"Min: {np.nanmin(self.data):.2f}, Max: {np.nanmax(self.data):.2f}")
        
        missing = np.sum(np.isnan(self.data))
        print(f"Missing values: {missing} ({missing/self.data.size*100:.2f}%)")