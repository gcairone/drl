from typing import List, Optional, Type
import torch 
import torch.nn as nn


"""
Base torch model module.
Implements a base class for PyTorch models with save/load utilities,
and a standard Deep Neural Network (DNN) implementation.
"""

class BaseTorchModel(nn.Module):
    """
    Base class for PyTorch models.
    """
    def __init__(self) -> None:
        super(BaseTorchModel, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.
        
        Args:
            x (torch.Tensor): Input tensor.
            
        Returns:
            torch.Tensor: Output tensor.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")

    def save(self, filepath: str) -> None:
        """
        Saves the model's state dictionary to a file.
        
        Args:
            filepath (str): Path to the file where the model will be saved.
        """
        torch.save(self.state_dict(), filepath)

    def load(self, filepath: str, map_location: Optional[str] = None) -> None:
        """
        Loads the model's state dictionary from a file.
        
        Args:
            filepath (str): Path to the file from which to load the model.
            map_location (str, optional): Device location map. Defaults to None.
        """
        self.load_state_dict(torch.load(filepath, map_location=map_location))

class BaseDNN(BaseTorchModel):
    """
    Standard Deep Neural Network (MLP) implementation.
    """
    def __init__(self, input_dim: int, output_dim: int, hidden_layers: List[int] = [64, 64], activation: Type[nn.Module] = nn.ReLU) -> None:
        super(BaseDNN, self).__init__()
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(activation())
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)