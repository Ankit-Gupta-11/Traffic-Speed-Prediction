import numpy as np
import torch


def mae(y_true, y_pred):
    """Mean Absolute Error."""
    return np.mean(np.abs(y_true - y_pred))


def rmse(y_true, y_pred):
    """Root Mean Square Error."""
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def mape(y_true, y_pred, threshold=5.0):
    """Mean Absolute Percentage Error."""
    # Avoid division by zero
    mask = np.abs(y_true) > threshold
    if not np.any(mask):
        return 0.0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def evaluate(model, data_loader, device='cpu', scaler=None):
    """
    Evaluate model on a dataset.    
    """
    model.eval()
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch = X_batch.to(device)
            
            predictions = model(X_batch)
            
            all_preds.append(predictions.cpu().numpy())
            all_targets.append(y_batch.numpy())
    
    # Concatenate all batches
    y_pred = np.concatenate(all_preds, axis=0)
    y_true = np.concatenate(all_targets, axis=0)
    
    # Inverse transform if scaler provided
    if scaler is not None:
        # Reshape for scaler
        orig_shape = y_pred.shape
        y_pred = scaler.inverse_transform(y_pred.reshape(-1, orig_shape[-1]))
        y_true = scaler.inverse_transform(y_true.reshape(-1, orig_shape[-1]))
        y_pred = y_pred.reshape(orig_shape)
        y_true = y_true.reshape(orig_shape)
    
    # Flatten for metrics
    y_pred_flat = y_pred.flatten()
    y_true_flat = y_true.flatten()
    
    # Calculate metrics
    metrics = {
        'mae': mae(y_true_flat, y_pred_flat),
        'rmse': rmse(y_true_flat, y_pred_flat),
        'mape': mape(y_true_flat, y_pred_flat)
    }
    
    return metrics, y_pred, y_true


def evaluate_horizons(model, data_loader, device='cpu', scaler=None, horizons=[3, 6, 12]):
    """
    Evaluate at different prediction horizons.
    """
    model.eval()
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch = X_batch.to(device)
            predictions = model(X_batch)
            all_preds.append(predictions.cpu().numpy())
            all_targets.append(y_batch.numpy())
    
    y_pred = np.concatenate(all_preds, axis=0)
    y_true = np.concatenate(all_targets, axis=0)
    
    # Inverse transform
    if scaler is not None:
        orig_shape = y_pred.shape
        y_pred = scaler.inverse_transform(y_pred.reshape(-1, orig_shape[-1]))
        y_true = scaler.inverse_transform(y_true.reshape(-1, orig_shape[-1]))
        y_pred = y_pred.reshape(orig_shape)
        y_true = y_true.reshape(orig_shape)
    
    # Evaluate at each horizon
    results = {}
    for h in horizons:
        if h > y_pred.shape[1]:
            continue
        
        pred_h = y_pred[:, :h, :].flatten()
        true_h = y_true[:, :h, :].flatten()
        
        results[f'{h*5}min'] = {
            'mae': mae(true_h, pred_h),
            'rmse': rmse(true_h, pred_h),
            'mape': mape(true_h, pred_h)
        }
    
    return results


def print_metrics(metrics, title="Evaluation Results"):
    """Pretty print metrics."""
    print("\n" + "=" * 40)
    print(title)
    print("=" * 40)
    
    if isinstance(metrics, dict) and 'mae' in metrics:
        # Single set of metrics
        print(f"MAE:  {metrics['mae']:.4f}")
        print(f"RMSE: {metrics['rmse']:.4f}")
        print(f"MAPE: {metrics['mape']:.2f}%")
    else:
        # Multiple horizons
        print(f"{'Horizon':<10} {'MAE':<10} {'RMSE':<10} {'MAPE':<10}")
        print("-" * 40)
        for horizon, m in metrics.items():
            print(f"{horizon:<10} {m['mae']:<10.4f} {m['rmse']:<10.4f} {m['mape']:<10.2f}%")
    
    print("=" * 40)