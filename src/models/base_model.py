import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple

class BaseModel(nn.Module, ABC):
    """
    Abstract base class for traffic prediction models.
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        input_steps: int,
        output_steps: int,
        device: Optional[torch.device] = None
    ):

        super(BaseModel, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_steps = input_steps
        self.output_steps = output_steps
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        print(
            f"Initialized {self.__class__.__name__}: "
            f"input_dim={input_dim}, output_dim={output_dim}, "
            f"input_steps={input_steps}, output_steps={output_steps}, "
            f"device={self.device}"
        )
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.
        """
        pass
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Make predictions (inference mode).
        """
        self.eval()
        with torch.no_grad():
            return self.forward(x)
    
    def get_num_parameters(self) -> Dict[str, int]:
        """
        Get the number of model parameters.
        """
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total': total,
            'trainable': trainable,
            'non_trainable': total - trainable
        }
    
    def save(self, filepath: str) -> None:
        """
        Save model weights and configuration.
        """
        state = {
            'model_state_dict': self.state_dict(),
            'config': {
                'input_dim': self.input_dim,
                'output_dim': self.output_dim,
                'input_steps': self.input_steps,
                'output_steps': self.output_steps,
                'model_class': self.__class__.__name__
            }
        }
        torch.save(state, filepath)
        print(f"Model saved to {filepath}")
    
    def load(self, filepath: str, strict: bool = True) -> None:
        """
        Load model weights from file.
        """
        state = torch.load(filepath, map_location=self.device)
        self.load_state_dict(state['model_state_dict'], strict=strict)
        print(f"Model loaded from {filepath}")
    
    def summary(self) -> str:
        """
        Get a summary of the model architecture.
        """
        params = self.get_num_parameters()
        
        lines = [
            "=" * 60,
            f"Model: {self.__class__.__name__}",
            "=" * 60,
            f"Input shape: (batch, {self.input_steps}, {self.input_dim})",
            f"Output shape: (batch, {self.output_steps}, {self.output_dim})",
            "-" * 60,
            f"Total parameters: {params['total']:,}",
            f"Trainable parameters: {params['trainable']:,}",
            f"Non-trainable parameters: {params['non_trainable']:,}",
            "-" * 60,
            "Layers:",
        ]
        
        for name, module in self.named_children():
            lines.append(f"  {name}: {module.__class__.__name__}")
        
        lines.append("=" * 60)
        
        return "\n".join(lines)
    
    def to_device(self, x: torch.Tensor) -> torch.Tensor:
        """
        Move tensor to model's device.
        """
        return x.to(self.device)
    
    def __repr__(self) -> str:
        params = self.get_num_parameters()
        return (
            f"{self.__class__.__name__}("
            f"input_dim={self.input_dim}, "
            f"output_dim={self.output_dim}, "
            f"params={params['trainable']:,})"
        )