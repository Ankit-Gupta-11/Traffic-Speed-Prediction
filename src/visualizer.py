import numpy as np
import matplotlib.pyplot as plt


def plot_training_history(train_losses, val_losses, save_path=None):
    """Plot training and validation loss curves."""
    plt.figure(figsize=(10, 5))
    
    epochs = range(1, len(train_losses) + 1)
    
    plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss')
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training History')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()


def plot_predictions(y_true, y_pred, sensor_idx=0, num_samples=3, save_path=None):
    """Plot actual vs predicted values."""
    fig, axes = plt.subplots(num_samples, 1, figsize=(12, 3*num_samples))
    
    if num_samples == 1:
        axes = [axes]
    
    # Random sample indices
    np.random.seed(42)
    sample_indices = np.random.choice(len(y_true), num_samples, replace=False)
    
    for i, (ax, idx) in enumerate(zip(axes, sample_indices)):
        actual = y_true[idx, :, sensor_idx]
        predicted = y_pred[idx, :, sensor_idx]
        timesteps = range(len(actual))
        
        ax.plot(timesteps, actual, 'b-o', label='Actual', markersize=5)
        ax.plot(timesteps, predicted, 'r--s', label='Predicted', markersize=5)
        
        ax.set_xlabel('Timestep')
        ax.set_ylabel('Speed (mph)')
        ax.set_title(f'Sample {idx}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()


def plot_error_distribution(y_true, y_pred, save_path=None):
    """Plot error distribution."""
    errors = (y_pred - y_true).flatten()
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Error histogram
    axes[0].hist(errors, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
    axes[0].axvline(x=0, color='red', linestyle='--', linewidth=2)
    axes[0].axvline(x=np.mean(errors), color='green', linestyle='--', linewidth=2, label=f'Mean: {np.mean(errors):.2f}')
    axes[0].set_xlabel('Error (Predicted - Actual)')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Error Distribution')
    axes[0].legend()
    
    # Scatter plot
    sample_size = min(5000, len(y_true.flatten()))
    indices = np.random.choice(len(y_true.flatten()), sample_size, replace=False)
    
    y_true_flat = y_true.flatten()[indices]
    y_pred_flat = y_pred.flatten()[indices]
    
    axes[1].scatter(y_true_flat, y_pred_flat, alpha=0.3, s=10)
    
    # Perfect prediction line
    min_val = min(y_true_flat.min(), y_pred_flat.min())
    max_val = max(y_true_flat.max(), y_pred_flat.max())
    axes[1].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
    
    axes[1].set_xlabel('Actual Speed (mph)')
    axes[1].set_ylabel('Predicted Speed (mph)')
    axes[1].set_title('Actual vs Predicted')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()