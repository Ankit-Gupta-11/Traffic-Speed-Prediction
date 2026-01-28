import torch
import torch.nn as nn


class LSTMModel(nn.Module):
    """Simple LSTM model for traffic prediction."""
    
    def __init__(self, input_size, hidden_size=64, num_layers=2, output_steps=12, dropout=0.2):

        super(LSTMModel, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_steps = output_steps
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Output layer
        self.fc = nn.Linear(hidden_size, input_size * output_steps)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """
        Forward pass.
        """
        batch_size = x.size(0)
        
        # LSTM forward
        lstm_out, _ = self.lstm(x)
        
        # Take last timestep output
        last_out = lstm_out[:, -1, :]  # (batch, hidden_size)
        
        # Apply dropout
        last_out = self.dropout(last_out)
        
        # Fully connected layer
        out = self.fc(last_out)  # (batch, features * output_steps)
        
        # Reshape to (batch, output_steps, features)
        out = out.view(batch_size, self.output_steps, self.input_size)
        
        return out
    
    def count_parameters(self):
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

def get_model(model_type, input_size, hidden_size=64, num_layers=2, output_steps=12, dropout=0.2):
    """Factory function to create models."""
    
    if model_type == 'lstm':
        model = LSTMModel(input_size, hidden_size, num_layers, output_steps, dropout)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    print(f"Created {model_type.upper()} model with {model.count_parameters():,} parameters")
    
    return model