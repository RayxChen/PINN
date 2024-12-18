import torch.nn as nn
import torch
import math

class PINN(nn.Module):
    """Physics-Informed Neural Network."""
    def __init__(self, input_dim=2, output_dim=1, hidden_layers=3, neurons_per_layer=64):
        super(PINN, self).__init__()
        layers = []

        # Input layer
        layers.append(nn.Linear(input_dim, neurons_per_layer))
        layers.append(nn.Tanh())

        # Hidden layers
        for _ in range(hidden_layers):
            layers.append(nn.Linear(neurons_per_layer, neurons_per_layer))
            layers.append(nn.Tanh())

        # Output layer
        layers.append(nn.Linear(neurons_per_layer, output_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, S, t, v=None):
        """
        Forward pass of the network.
        Args:
            S: Stock price
            t: Time
            v: Volatility (optional, for Heston model)
        """
        if v is None:
            # Black-Scholes case
            if S.dim() == 1:
                S = S.unsqueeze(1)
            if t.dim() == 1:
                t = t.unsqueeze(1)
            x = torch.cat([S, t], dim=1)
        else:
            # Heston case
            if S.dim() == 1:
                S = S.unsqueeze(1)
            if t.dim() == 1:
                t = t.unsqueeze(1)
            if v.dim() == 1:
                v = v.unsqueeze(1)
            x = torch.cat([S, t, v], dim=1)
        return self.net(x)


class FinancialPINN(nn.Module):
    """Physics-Informed Neural Network for Financial Applications."""
    
    def __init__(self, input_dim=2, hidden_dim=64, hidden_layers=4, fourier_features=50, omega_0=30):
        """
        Args:
            input_dim (int): Number of input features (e.g., S, K, T).
            hidden_dim (int): Number of neurons in each hidden layer.
            hidden_layers (int): Number of residual blocks in the network.
            fourier_features (int): Number of Fourier features to use.
            omega_0 (float): Scaling factor for SIREN-like activation functions.
        """
        super(FinancialPINN, self).__init__()
        
        # Fourier feature mapping
        self.B = torch.randn((input_dim, fourier_features)) * 2 * math.pi  # Random Fourier frequencies
        self.fourier_dim = 2 * fourier_features  # Fourier features output dimension (sin + cos)

        # Input layer
        self.input_layer = nn.Linear(self.fourier_dim, hidden_dim)

        # Residual blocks
        self.residual_blocks = nn.ModuleList([
            self.ResidualBlock(hidden_dim, hidden_dim) for _ in range(hidden_layers)
        ])

        # Skip connection layers
        self.skip_layers = nn.ModuleList([
            nn.Linear(self.fourier_dim, hidden_dim) 
            for _ in range((hidden_layers + 1) // 2)  # Use ceiling division
        ])

        # Output layer
        self.output_layer = nn.Linear(hidden_dim, 1)  # Predict option price or volatility

        # SIREN-like scaling factor
        self.omega_0 = omega_0

    def forward(self, S, t):
        """
        Forward pass for the network.

        Args:
            inputs (torch.Tensor): Input tensor (e.g., [S, K, T]).

        Returns:
            torch.Tensor: Predicted option price or volatility.
        """
        inputs = torch.cat([S, t], dim=1)
        # Fourier feature mapping
        inputs = self.apply_fourier_features(inputs)

        # Input transformation
        x = self.input_layer(inputs)
        x = torch.sin(self.omega_0 * x)  # SIREN-like activation

        # Pass through residual blocks with skip connections
        for i, block in enumerate(self.residual_blocks):
            x = block(x)
            if i % 2 == 0:  # Add skip connection every two blocks
                skip = self.skip_layers[i // 2](inputs)
                x += skip

        # Output transformation
        output = self.output_layer(x)
        return output

    def apply_fourier_features(self, x):
        """
        Apply Fourier feature mapping to the input.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim).

        Returns:
            torch.Tensor: Fourier-transformed input.
        """
        x_proj = torch.matmul(x, self.B.to(x.device))  # Project input onto Fourier basis
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

    class ResidualBlock(nn.Module):
        """Residual block with skip connection."""
        def __init__(self, input_dim, hidden_dim):
            """
            Args:
                input_dim (int): Input feature dimension.
                hidden_dim (int): Hidden layer feature dimension.
            """
            super(FinancialPINN.ResidualBlock, self).__init__()
            self.fc1 = nn.Linear(input_dim, hidden_dim)
            self.activation = nn.Tanh()  # You can replace with SIREN-like activation if needed
            self.fc2 = nn.Linear(hidden_dim, input_dim)

        def forward(self, x):
            """
            Forward pass for the residual block.

            Args:
                x (torch.Tensor): Input tensor.

            Returns:
                torch.Tensor: Output tensor with skip connection applied.
            """
            residual = x
            x = self.fc1(x)
            x = self.activation(x)
            x = self.fc2(x)
            return x + residual  # Skip connection