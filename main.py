import os
import numpy as np
import torch

from config import config
from src.data_loader import DataLoader
from src.preprocessor import Preprocessor, split_data, create_sequences
from src.dataset import get_dataloaders
from src.models.lstm_model import get_model
from src.trainer import Trainer
from src.evaluator import evaluate, evaluate_horizons, print_metrics
from src.visualizer import plot_training_history, plot_predictions, plot_error_distribution


def main():
    # Set random seed
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Create results directory
    os.makedirs(config['results_dir'], exist_ok=True)
    
    print("Traffic Speed Prediction using LSTM")
    
    # Step 1: Load Data
    print("\n[Step 1] Loading data...")
    
    loader = DataLoader(config['data_path'])
    data, timestamps = loader.load()
    
    # Use subset of sensors if specified
    if config['num_sensors'] is not None:
        num_sensors = min(config['num_sensors'], data.shape[1])
        print(f"Using first {num_sensors} sensors")
        data = data[:, :num_sensors]
    
    num_sensors = data.shape[1]
    
    # Step 2: Preprocess Data
    print("\n[Step 2] Preprocessing data...")
    
    # Split data
    train_data, val_data, test_data = split_data(
        data, 
        train_ratio=config['train_ratio'], 
        val_ratio=config['val_ratio']
    )
    
    # Normalize
    preprocessor = Preprocessor()
    preprocessor.fit(train_data)
    
    train_norm = preprocessor.transform(train_data)
    val_norm = preprocessor.transform(val_data)
    test_norm = preprocessor.transform(test_data)
    
    # Step 3: Create Sequences
    print("\n[Step 3] Creating sequences...")
    
    input_len = config['input_len']
    output_len = config['output_len']
    
    X_train, y_train = create_sequences(train_norm, input_len=input_len, output_len=output_len)
    X_val, y_val = create_sequences(val_norm, input_len=input_len, output_len=output_len)
    X_test, y_test = create_sequences(test_norm, input_len=input_len, output_len=output_len)
    
    # Step 4: Create DataLoaders
    print("\n[Step 4] Creating data loaders...")
    
    train_loader, val_loader, test_loader = get_dataloaders(
        X_train, y_train, X_val, y_val, X_test, y_test,
        batch_size=config['batch_size']
    )
    
    # Step 5: Create Model
    print("\n[Step 5] Creating model...")
    
    model = get_model(
        model_type=config['model_type'],
        input_size=num_sensors,
        hidden_size=config['hidden_size'],
        num_layers=config['num_layers'],
        output_steps=output_len,
        dropout=config['dropout']
    )
    
    # Step 6: Train Model
    print("\n[Step 6] Training model...")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    trainer = Trainer(model, device=device, lr=config['learning_rate'])
    train_losses, val_losses = trainer.train(
        train_loader, val_loader,
        epochs=config['epochs'],
        patience=config['patience']
    )
    
    # Save model
    model_path = os.path.join(config['results_dir'], 'best_model.pth')
    trainer.save_model(model_path)
    
    # Plot training history
    history_path = os.path.join(config['results_dir'], 'training_history.png')
    plot_training_history(train_losses, val_losses, save_path=history_path)
    
    # Step 7: Evaluate Model
    print("\n[Step 7] Evaluating model...")
    
    # Overall metrics
    metrics, y_pred, y_true = evaluate(model, test_loader, device=device, scaler=preprocessor)
    print_metrics(metrics, title="Test Set Results")
    
    # Per-horizon metrics
    horizon_results = evaluate_horizons(
        model, test_loader, device=device, scaler=preprocessor,
        horizons=[3, 6, 12]  # 15min, 30min, 60min
    )
    print_metrics(horizon_results, title="Results by Prediction Horizon")
    
    # Step 8: Visualize Results
    print("\n[Step 8] Generating visualizations...")
    
    pred_path = os.path.join(config['results_dir'], 'predictions.png')
    error_path = os.path.join(config['results_dir'], 'error_distribution.png')
    
    plot_predictions(y_true, y_pred, sensor_idx=0, num_samples=3, save_path=pred_path)
    plot_error_distribution(y_true, y_pred, save_path=error_path)
    
    print("\n" + "=" * 50)
    print(f"Done! Results saved to '{config['results_dir']}/' folder")
    print("=" * 50)


if __name__ == "__main__":
    main()