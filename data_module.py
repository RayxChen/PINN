import torch
import numpy as np
from solver import BlackScholes, Heston
from data_generator import GeometricBrownianMotion, HestonModel

# def generate_synthetic_data(config):
#     """
#     Generate synthetic data for PINN using GBM and Black-Scholes formula.

#     Parameters:
#         config (dict): Configuration dictionary containing:
#             - S0: Initial stock price.
#             - mu: Drift (expected return).
#             - sigma: Volatility.
#             - T: Time to maturity.
#             - num_paths: Number of simulated paths.
#             - num_steps: Number of time steps.
#             - K: Strike price.
#             - r: Risk-free interest rate.

#     Returns:
#         S_tensor (torch.Tensor): Stock prices (inputs for PINN).
#         t_tensor (torch.Tensor): Time points (inputs for PINN).
#         V_tensor (torch.Tensor): Option prices (targets for supervised learning).
#     """
#     # Generate synthetic stock price paths
#     t, paths = GeometricBrownianMotion(
#         S0=config["model"]["S0"],
#         mu=config["model"]["mu"],
#         sigma=config["model"]["sigma"],
#         T=config["model"]["T"],
#         num_paths=config["simulation"]["num_paths"],
#         num_steps=config["simulation"]["num_steps"]
#     ).generate_paths()

#     # Flatten S and t for PINN input
#     S_flat = paths.flatten()  # Flatten stock prices
#     t_flat = np.repeat(t, config["simulation"]["num_paths"])  # Repeat time for each path

#     # Compute theoretical prices using Black-Scholes formula
#     V_flat = np.array([
#         BlackScholes(S, config["model"]["K"], T_remaining, config["model"]["r"], config["model"]["sigma"], option_type="call")
#         for S, T_remaining in zip(S_flat, config["model"]["T"] - t_flat)
#     ])

#     # Convert to PyTorch tensors
#     S_tensor = torch.tensor(S_flat, dtype=torch.float32).view(-1, 1)
#     t_tensor = torch.tensor(t_flat, dtype=torch.float32).view(-1, 1)
#     V_tensor = torch.tensor(V_flat, dtype=torch.float32).view(-1, 1)

#     return S_tensor, t_tensor, V_tensor


def generate_synthetic_data(config):
    """
    Generate synthetic data for PINN using GBM or Heston model.

    Parameters:
        config (dict): Configuration dictionary.

    Returns:
        S_tensor (torch.Tensor): Stock prices (inputs for PINN).
        t_tensor (torch.Tensor): Time points (inputs for PINN).
        v_tensor (torch.Tensor, optional): Variance paths for Heston model.
        V_tensor (torch.Tensor): Option prices (targets for supervised learning).
    """
    model_type = config["model_type"]

    if model_type == "Heston":
        # Generate Heston model paths
        t, S_paths, v_paths = HestonModel(
            S0=config["model"]["S0"],
            v0=config["model"]["v0"],
            kappa=config["model"]["kappa"],
            theta=config["model"]["theta"],
            sigma_v=config["model"]["sigma_v"],
            rho=config["model"]["rho"],
            r=config["model"]["r"],
            T=config["model"]["T"],
            num_paths=config["simulation"]["num_paths"],
            num_steps=config["simulation"]["num_steps"]
        ).generate_paths()

        # Flatten stock prices, variance, and time
        S_flat = S_paths.flatten()
        v_flat = v_paths.flatten()
        t_flat = np.repeat(t, config["simulation"]["num_paths"])

        # Add a small positive number to ensure strictly positive values
        epsilon = 1e-5
        S_flat = np.maximum(S_flat, epsilon)  # Ensure positive stock prices
        v_flat = np.maximum(v_flat, epsilon)  # Ensure positive variance
        
        # Compute theoretical prices
        V_flat = np.array([
            Heston(S, T_remaining, v, config)
            for S, T_remaining, v in zip(S_flat, config["model"]["T"] - t_flat, v_flat)
        ])

        # Convert to PyTorch tensors
        S_tensor = torch.tensor(S_flat, dtype=torch.float32).view(-1, 1)
        t_tensor = torch.tensor(t_flat, dtype=torch.float32).view(-1, 1)
        v_tensor = torch.tensor(v_flat, dtype=torch.float32).view(-1, 1)
        V_tensor = torch.tensor(V_flat, dtype=torch.float32).view(-1, 1)

        return S_tensor, t_tensor, v_tensor, V_tensor

    else:  # Default to GBM
        t, S_paths = GeometricBrownianMotion(
            S0=config["model"]["S0"],
            mu=config["model"]["mu"],
            sigma=config["model"]["sigma"],
            T=config["model"]["T"],
            num_paths=config["simulation"]["num_paths"],
            num_steps=config["simulation"]["num_steps"]
        ).generate_paths()

        # Flatten stock prices and time
        S_flat = S_paths.flatten()
        t_flat = np.repeat(t, config["simulation"]["num_paths"])

        # Compute theoretical Black-Scholes prices
        V_flat = np.array([
            BlackScholes(S, T_remaining, config)
            for S, T_remaining in zip(S_flat, config["model"]["T"] - t_flat)
        ])

        # Convert to PyTorch tensors
        S_tensor = torch.tensor(S_flat, dtype=torch.float32).view(-1, 1)
        t_tensor = torch.tensor(t_flat, dtype=torch.float32).view(-1, 1)
        V_tensor = torch.tensor(V_flat, dtype=torch.float32).view(-1, 1)

        return S_tensor, t_tensor, V_tensor


def generate_training_data(config, device):
    """Generate training data based on configuration."""
    model_type = config["model_type"]

    if config["training"]["use_data"]:
        print("Training with data-assisted PINN...")

        # Generate synthetic data
        if model_type == "Heston":
            S_data, t_data, v_data, V_data = generate_synthetic_data(config)
            S_data = S_data.to(device)
            t_data = t_data.to(device)
            v_data = v_data.to(device)
            V_data = V_data.to(device)

            dataset = torch.utils.data.TensorDataset(S_data, t_data, v_data, V_data)
        else:  # Black-Scholes
            S_data, t_data, V_data = generate_synthetic_data(config)
            S_data = S_data.to(device)
            t_data = t_data.to(device)
            V_data = V_data.to(device)

            dataset = torch.utils.data.TensorDataset(S_data, t_data, V_data)

    else:
        print("Training with pure PINN (no data)...")

        if model_type == "BlackScholes":
            # Generate grid points for Black-Scholes
            S_data, t_data = generate_grid_points(
                K=config["model"]["K"],
                T=config["model"]["T"],
                num_points=config["training"]["num_points"],
                device=device,
                model_type=model_type
            )
            dataset = torch.utils.data.TensorDataset(S_data, t_data)

        elif model_type == "Heston":
            # Generate grid points for Heston
            S_data, t_data, v_data = generate_grid_points(
                K=config["model"]["K"],
                T=config["model"]["T"],
                num_points=config["training"]["num_points"],
                device=device,
                model_type=model_type
            )
            dataset = torch.utils.data.TensorDataset(S_data, t_data, v_data)

        else:
            raise ValueError(f"Unknown model type: {model_type}")

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True
    )

    return dataloader

    
# def generate_training_data(config, device):
#     """Generate training data based on configuration."""
#     model_type = config["model_type"]
    
#     if config["training"]["use_data"]:
#         print("Training with data-assisted PINN...")
#         S_data, t_data, V_data = generate_synthetic_data(config)
#         S_data = S_data.to(device)
#         t_data = t_data.to(device)
#         V_data = V_data.to(device)
        
#         dataset = torch.utils.data.TensorDataset(S_data, t_data, V_data)
#     else:
#         print("Training with pure PINN (no data)...")
#         if model_type == "BlackScholes":
#             S_data, t_data = generate_grid_points(
#                 K=config["model"]["K"],
#                 T=config["model"]["T"],
#                 num_points=config["training"]["num_points"],
#                 device=device,
#                 model_type=model_type
#             )
#             dataset = torch.utils.data.TensorDataset(S_data, t_data)
            
#         elif model_type == "Heston":
#             S_data, t_data, v_data = generate_grid_points(
#                 K=config["model"]["K"], 
#                 T=config["model"]["T"],
#                 num_points=config["training"]["num_points"],
#                 device=device,
#                 model_type=model_type
#             )
#             dataset = torch.utils.data.TensorDataset(S_data, t_data, v_data)
#         else:
#             raise ValueError(f"Unknown model type: {model_type}")

#     dataloader = torch.utils.data.DataLoader(
#         dataset,
#         batch_size=config["training"]["batch_size"],
#         shuffle=True
#     )
    
#     return dataloader


def generate_grid_points(K, T, num_points, S_upper_bound=2, device=None, model_type="BlackScholes", epsilon=1e-3, uniform=True):
    if uniform:
        S = torch.linspace(epsilon, S_upper_bound * K, num_points).view(-1, 1)
    else:
        window = 0.2 * K
        S_concentrated = torch.linspace(K - window, K + window, num_points // 2)
        S_lower = torch.linspace(epsilon, K - window, num_points // 4)
        S_upper = torch.linspace(K + window, S_upper_bound * K, num_points // 4)
        S = torch.cat([S_lower, S_concentrated, S_upper]).view(-1, 1)
    
    t = torch.linspace(0, T, num_points).view(-1, 1)
    S_grid, t_grid = torch.meshgrid(S.flatten(), t.flatten(), indexing='ij')
    
    S_flat = S_grid.flatten().view(-1, 1)
    t_flat = t_grid.flatten().view(-1, 1)
    
    if device:
        S_flat, t_flat = S_flat.to(device), t_flat.to(device)
    
    if model_type == "Heston":
        # Sample variance using a Gamma distribution
        alpha, beta = 2.0, 2.0
        v = torch.distributions.Gamma(alpha, beta).sample([len(S_flat)]).view(-1, 1)
        if device:
            v = v.to(device)
        return S_flat, t_flat, v

    return S_flat, t_flat


