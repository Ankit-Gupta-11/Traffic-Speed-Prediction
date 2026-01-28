config = {
    # Data settings
    "data_path": "data/METR-LA.h5",
    "num_sensors": None,  # None = use all sensors
    
    # Sequence settings
    "input_len": 12,
    "output_len": 12,
    
    # Split ratios
    "train_ratio": 0.7,
    "val_ratio": 0.15,
    "test_ratio": 0.15,
    
    # Model settings
    "model_type": "lstm",
    "hidden_size": 64,
    "num_layers": 2,
    "dropout": 0.2,
    
    # Training settings
    "batch_size": 64,
    "epochs": 50,
    "learning_rate": 0.001,
    "patience": 15, # Early stopping patience
    
    # Output Dir
    "results_dir": "results",
}