# Traffic Speed Prediction using LSTM

A deep learning project for short-term traffic speed prediction using LSTM networks on the METR-LA dataset.

---

## Problem Statement

Traffin speed prediction is the task of forecasting future traffic condition based on historical observations. This project aims to:

- **Input**: 12 timesteps (1 hour) of historical traffic speeds
- **Output**: 3 timesteps (15 minutes) of predicted traffic speeds
- **Goal**: Minimize prediction error (MAE, RMSE, MAPE)

This is useful for traffic management systems, route planning, and congestion prediction.

---

## Dataset

### METR-LA (Los Angeles Metropolitan Traffic)

Traffic speed data collected from loop detectors on Los Angeles County highways.

| Attribute | Value |
|-----------|-------|
| Source | LA County loop detectors |
| Sensors | 207 |
| Time Period | March 1 - June 30, 2012 |
| Time Interval | 5 minutes |
| Total Timesteps | 34,272 |
| Data Type | Traffic speed (mph) |

**Download**: [Kaggle - METR-LA Dataset](https://www.kaggle.com/datasets/annnnguyen/metr-la-dataset)

### Key Characteristics (from EDA)

- **Rush Hours**: Speeds drop 30-40% during 7-9 AM and 5-7 PM
- **Weekend Effect**: Average speeds ~10-15% higher on weekends
- **Missing Values**: ~0.3% (handled via interpolation)
- **Speed Range**: 5-85 mph, mean ~55 mph

---

## Directory Structure

```
traffic_prediction/
│
├── config.py              # All hyperparameters and settings
├── main.py                # Main entry point - runs the full pipeline
├── requirements.txt       # Python dependencies
├── README.md              # This file
├── data/
│   └── METR-LA.h5         # Dataset (download separately)
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py     # Loads data from HDF5 file
│   ├── preprocessor.py    # Normalization, splitting, sequence creation
│   ├── dataset.py         # PyTorch Dataset and DataLoader
│   ├── trainer.py         # Training loop with early stopping
│   ├── evaluator.py       # Evaluation metrics (MAE, RMSE, MAPE)
│   ├── visualizer.py      # Plotting functions
│   └── models/
│       ├── __init__.py
│       ├── base_model.py  # Abstract class for ML model
│       └── lstm_model.py  # LSTM and BiLSTM models
│
└── results/
    ├── best_model.pth         # Saved model weights
    ├── training_history.png   # Training/validation loss plot
    ├── predictions.png        # Sample predictions visualization
    └── error_distribution.png # Error analysis plots
```

---

## Installation

### Requirements

- Python 3.12
- Refer `requirements.txt`

### Setup

```bash
# Clone or extract the project
cd traffic_prediction

# Install dependencies
pip install -r requirements.txt

# Download METR-LA.h5 from Kaggle and place in data/ folder
```

---

## Usage

### Running the Model

```bash
python main.py
```

This will:
1. Load data from `data/METR-LA.h5`
2. Preprocess (normalize, split, create sequences)
3. Train LSTM model with early stopping
4. Evaluate on test set
5. Save results to `results/` folder



---

## Model Architecture

### LSTM Network

![LSTM Network](report/Traffic%20Speed%20Prediction.png)

### Why LSTM?

- Captures temporal dependencies in sequential data
- Handles long-term patterns (rush hours, daily cycles)
- Well-suited for time series forecasting

---

## Results

### Training Summary

| Metric | Value |
|--------|-------|
| Total Epochs | 23 (early stopped) |
| Best Epoch | 8 |
| Best Val Loss | 0.3429 |
| Training Time | ~2 minutes (CPU) |

### Test Set Performance

| Model | MAE (mph) | RMSE (mph) | MAPE (%) |
|-------|-----------|------------|----------|
| Historical Average | ~7.8 | ~14.3 | ~15.0 |
| **LSTM (mine)** | **6.43** | **12.19** | **12.1** |
| DCRNN (State-of-art) | ~2.8 | ~5.4 | ~7.3 |

### Key Observations

1. **LSTM beats historical average** - Shows the model learned meaningful patterns
2. **High RMSE relative to MAE** - Indicates larger errors during rush hours
3. **Gap to state-of-art** - Graph-based methods (DCRNN) perform significantly better as per the papers

---

## Pipeline Overview

```
1. Data Loading
   └── Load HDF5 file, extract speed matrix (34272 × 207)

2. Preprocessing
   ├── Handle missing values (interpolation)
   ├── Split: 70% train, 15% val, 15% test (temporal split)
   ├── Z-score normalization (fit on train only)
   └── Create sliding window sequences (X: 12 steps, y: 3 steps)

3. Model Training
   ├── LSTM forward pass
   ├── MSE loss computation
   ├── Adam optimizer
   ├── Gradient clipping (max_norm=5)
   └── Early stopping (patience=15)

4. Evaluation
   ├── Compute MAE, RMSE, MAPE
   ├── Inverse transform predictions
   └── Compare with baselines

5. Visualization
   ├── Training history plot
   ├── Prediction samples
   └── Error distribution
```

---

## Evaluation Metrics

| Metric | Formula | Description |
|--------|---------|-------------|
| **MAE** | Mean(\|y - ŷ\|) | Average absolute error in mph |
| **RMSE** | √Mean((y - ŷ)²) | Penalizes large errors more |
| **MAPE** | Mean(\|y - ŷ\| / y) × 100 | Percentage error |

---

## Limitations

1. **No spatial modeling** - Each sensor is treated independently
2. **Rush hour difficulty** - High variance periods are harder to predict
3. **Fixed prediction horizon** - Only predicts 15 minutes ahead
4. **No external features** - Doesn't use weather, events, holidays

---

## Future Improvements

| Improvement | Description | Expected Impact |
|-------------|-------------|-----------------|
| Time features | Add hour, day_of_week as inputs | Better rush hour handling |
| BiLSTM | Bidirectional LSTM | Capture forward/backward patterns |
| Attention | Focus on important timesteps | Improved accuracy |
| GNN | Model sensor relationships | Significant improvement |
| Multi-horizon | Predict 15/30/60 min together | More flexible |

---

## References

**Dataset**
- [METR-LA on Kaggle](https://www.kaggle.com/datasets/annnnguyen/metr-la-dataset)

**Resources**
- Li et al. (2018) - "Diffusion Convolutional Recurrent Neural Network" - [arXiv](https://arxiv.org/abs/1707.01926)
- [Kaggle Notebook](https://www.kaggle.com/code/annnnguyen/trafficforecastinggnn)
- [Traffin Flow Prediction](https://github.com/thenomaniqbal/Traffic-flow-prediction)
- [Traffic Flow Prediction System Guide](https://medium.com/@devarshpatel15062001/building-an-end-to-end-traffic-flow-prediction-system-a-step-by-step-guide-3f201c7a9c9f)