import torch 


"""
Base torch model, implements torch module, has model architecture, save/load methods.
"""

class BaseTorchModel(torch.nn.Module):
    def __init__(self):
        super(BaseTorchModel, self).__init__()

    def forward(self, x):
        raise NotImplementedError("This method should be overridden by subclasses.")

    def save(self, filepath):
        torch.save(self.state_dict(), filepath)

    def load(self, filepath, map_location=None):
        self.load_state_dict(torch.load(filepath, map_location=map_location))

class BaseDNN(BaseTorchModel):
    """
    Base Deep Neural Network model.
    """
    def __init__(self, input_dim, output_dim, hidden_layers=[64, 64], activation=torch.nn.ReLU):
        super(BaseDNN, self).__init__()
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_layers:
            layers.append(torch.nn.Linear(prev_dim, hidden_dim))
            layers.append(activation())
            prev_dim = hidden_dim
        layers.append(torch.nn.Linear(prev_dim, output_dim))
        self.network = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)