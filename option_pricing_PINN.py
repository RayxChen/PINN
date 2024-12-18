import os
import json
from abc import ABC
import numpy as np
import torch
from torch import optim
from solver import BlackScholes, Heston
from data_module import generate_training_data, generate_grid_points
from Greeks import GreeksCalculator
from network import PINN
from plots import plot_comparison
from utils import create_directories
from torch.utils.tensorboard import SummaryWriter


class OptionPricingPINN(ABC):
    def __init__(self, config, device):
        self.config = config
        self.device = device

    def compute_boundary_loss(self, *args, **kwargs):
        pass

    def compute_pde_residual(self, *args, **kwargs):
        pass


class BlackScholesSolver(OptionPricingPINN):
    def __init__(self, config, device, model):
        super().__init__(config, device)
        self.model = model

    def compute_boundary_loss(self, S, t):
        K = self.config["model"]["K"]
        T = self.config["model"]["T"]
        S_upper_bound = self.config["model"]["S_upper_bound"]

        # Terminal Condition
        t_terminal = torch.ones_like(t) * T
        terminal_condition = self.model(S, t_terminal)
        terminal_loss = torch.mean((terminal_condition - torch.max(S - K, torch.zeros_like(S))) ** 2)

        # Lower Boundary: V(0, t) = 0
        S_lower = torch.zeros_like(t)
        lower_boundary_loss = torch.mean(self.model(S_lower, t) ** 2)

        # Upper Boundary: V(S -> large, t) = S - K
        S_upper = torch.ones_like(t) * (S_upper_bound * K)
        upper_boundary_loss = torch.mean((self.model(S_upper, t) - (S_upper - K)) ** 2)

        return terminal_loss + lower_boundary_loss + upper_boundary_loss

    def compute_pde_residual(self, S, t, V):
        S.requires_grad = True
        t.requires_grad = True
        grad_outputs = torch.ones_like(V).to(self.device)

        V_t, V_S = torch.autograd.grad(V, [t, S], grad_outputs=(grad_outputs, grad_outputs),
                                       create_graph=True, retain_graph=True)
        V_SS = torch.autograd.grad(V_S, S, grad_outputs=grad_outputs, create_graph=True)[0]

        sigma = self.config["model"]["sigma"]
        r = self.config["model"]["r"]
        pde_residual = V_t + 0.5 * sigma ** 2 * S ** 2 * V_SS + r * S * V_S - r * V

        return pde_residual

    @staticmethod
    def compute_solution(S, T, config):
        return BlackScholes(S, T, config)

class HestonSolver(OptionPricingPINN):
    def __init__(self, config, device, model):
        super().__init__(config, device)
        self.model = model

    def compute_boundary_loss(self, S, t, v):
        K = self.config["model"]["K"]
        T = self.config["model"]["T"]
        S_upper_bound = self.config["model"]["S_upper_bound"]

        # Terminal Condition
        t_terminal = torch.ones_like(t) * T
        terminal_condition = self.model(S, t_terminal, v)
        terminal_loss = torch.mean((terminal_condition - torch.max(S - K, torch.zeros_like(S))) ** 2)

        # Lower Boundary: V(0, t) = 0
        S_lower = torch.zeros_like(t)
        lower_boundary_loss = torch.mean(self.model(S_lower, t, v) ** 2)

        # Upper Boundary: V(S -> ∞, t) = S - K*e^(-r(T-t))
        S_upper = torch.ones_like(t) * (S_upper_bound * K)
        r = self.config["model"]["r"]
        T = self.config["model"]["T"]
        discounted_K = K * torch.exp(-r * (T - t))
        upper_boundary_loss = torch.mean((self.model(S_upper, t, v) - (S_upper - discounted_K)) ** 2)

        return terminal_loss + lower_boundary_loss + upper_boundary_loss

    def compute_pde_residual(self, S, t, v, V):
        S.requires_grad = True
        t.requires_grad = True
        v.requires_grad = True

        grad_outputs = torch.ones_like(V).to(self.device)
        
        # First-order derivatives
        V_t, V_S, V_v = torch.autograd.grad(V, [t, S, v], grad_outputs=(grad_outputs, grad_outputs, grad_outputs),
                                        create_graph=True, retain_graph=True)
        
        # Second-order derivatives
        V_SS = torch.autograd.grad(V_S, S, grad_outputs=grad_outputs, create_graph=True)[0]
        V_vv = torch.autograd.grad(V_v, v, grad_outputs=grad_outputs, create_graph=True)[0]
        V_Sv = torch.autograd.grad(V_S, v, grad_outputs=grad_outputs, create_graph=True)[0]  # Mixed derivative

        # Extract parameters
        kappa = self.config["model"]["kappa"]  # Mean-reversion speed
        theta = self.config["model"]["theta"]  # Long-run variance
        sigma_v = self.config["model"]["sigma_v"]  # Volatility of variance
        rho = self.config["model"]["rho"]  # Correlation between S and v
        r = self.config["model"]["r"]  # Risk-free rate

        # Heston PDE residual
        pde_residual = (V_t + r * S * V_S - r * V +
                        0.5 * S**2 * v * V_SS +
                        rho * sigma_v * S * v**0.5 * V_Sv +  # Mixed derivative term
                        0.5 * sigma_v**2 * v * V_vv +
                        kappa * (theta - v) * V_v)
        
        return pde_residual

    @staticmethod
    def compute_solution(S, t, v, config):
        return Heston(S, t, v, config)


class OptionPricingSolver:
    """
    Handles the setup, training, and evaluation for the option pricing PINN.
    Supports multiple models (Black-Scholes, Heston).
    """
    def __init__(self, config, load_checkpoint=None):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # Create or load directories
        self.save_dir, self.checkpoint_dir, self.plot_dir, self.tensorboard_dir = create_directories(config, load_checkpoint)
        self.writer = SummaryWriter(log_dir=self.tensorboard_dir)

        # Save config
        with open(os.path.join(self.save_dir, "config.json"), "w") as f:
            json.dump(config, f, indent=4)

        # Initialise model and optimizer
        self.model = PINN(
            input_dim=config["network"]["input_dim"],
            hidden_layers=config["network"]["hidden_layers"],
            neurons_per_layer=config["network"]["neurons_per_layer"]
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config["training"]["learning_rate"])
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=5)
        self.dataloader = generate_training_data(config, self.device)

        # Load checkpoint if specified
        if load_checkpoint:
            self._load_checkpoint(load_checkpoint)

        # Select model type
        self.model_type = config.get("model_type", "BlackScholes")
        if self.model_type == "Heston":
            self.solver = HestonSolver(config, self.device, self.model)
        else:
            self.solver = BlackScholesSolver(config, self.device, self.model)
        
        self.greeks = GreeksCalculator(self)

    def _load_checkpoint(self, checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        print("Checkpoint loaded successfully.")

    def loss_function(self, S, t, v=None, V_true=None, return_components=False):
        S.requires_grad = True
        t.requires_grad = True
        if self.model_type == "Heston":
            v.requires_grad = True
            V = self.model(S, t, v)
            pde_residual = self.solver.compute_pde_residual(S, t, v, V)
            boundary_loss = self.solver.compute_boundary_loss(S, t, v)
        else:
            V = self.model(S, t)
            pde_residual = self.solver.compute_pde_residual(S, t, V)
            boundary_loss = self.solver.compute_boundary_loss(S, t)

        pde_loss = torch.mean(pde_residual ** 2)

        data_loss = torch.tensor(0.0, device=self.device)
        if self.config["training"]["use_data"] and V_true is not None:
            data_loss = torch.mean((V - V_true) ** 2)

        total_loss = (self.config["weights"]["pde_weight"] * pde_loss +
                      self.config["weights"]["boundary_weight"] * boundary_loss +
                      self.config["weights"]["data_weight"] * data_loss)

        if return_components:
            return total_loss, {'pde_loss': pde_loss.item(),
                                'boundary_loss': boundary_loss.item(),
                                'data_loss': data_loss.item()}
        return total_loss

    def train(self):
        """Train the model using either pure PINN or data-assisted PINN approach."""
        losses = []
        for epoch in range(self.config["training"]["epochs"]):
            epoch_loss = 0
            pde_losses = 0
            boundary_losses = 0
            data_losses = 0

            for batch in self.dataloader:
                # Initialize V_batch as None by default
                V_batch = None
                
                if self.model_type == "Heston":
                    if self.config["training"]["use_data"]:
                        S_batch, t_batch, v_batch, V_batch = batch  # Unpack 4 values for data-assisted Heston
                    else:
                        S_batch, t_batch, v_batch = batch  # Unpack 3 values for pure Heston PINN
                else:  # Black-Scholes
                    if self.config["training"]["use_data"]:
                        S_batch, t_batch, V_batch = batch
                    else:
                        S_batch, t_batch = batch

                S_batch = S_batch.to(self.device)
                t_batch = t_batch.to(self.device)
                if V_batch is not None:
                    V_batch = V_batch.to(self.device)
                if self.model_type == "Heston":
                    v_batch = v_batch.to(self.device)
                    loss, loss_components = self.loss_function(
                        S_batch, t_batch, v=v_batch, V_true=V_batch, return_components=True
                    )
                else:
                    loss, loss_components = self.loss_function(
                        S_batch, t_batch, V_true=V_batch, return_components=True
                    )

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                pde_losses += loss_components['pde_loss']
                boundary_losses += loss_components['boundary_loss']
                data_losses += loss_components['data_loss']

            avg_loss = epoch_loss / len(self.dataloader)
            avg_pde_loss = pde_losses / len(self.dataloader)
            avg_boundary_loss = boundary_losses / len(self.dataloader)
            avg_data_loss = data_losses / len(self.dataloader)
            
            # Log losses to TensorBoard
            self.writer.add_scalar('Loss/total', avg_loss, epoch)
            self.writer.add_scalar('Loss/pde', avg_pde_loss, epoch)
            self.writer.add_scalar('Loss/boundary', avg_boundary_loss, epoch)
            self.writer.add_scalar('Loss/data', avg_data_loss, epoch)
            
            # Log learning rate
            self.writer.add_scalar('Learning_rate', 
                                 self.optimizer.param_groups[0]['lr'], 
                                 epoch)

            losses.append(avg_loss)

            if epoch % 500 == 0:
                print(f"Epoch {epoch}/{self.config['training']['epochs']}, "
                      f"Loss: {avg_loss:.6f}, "
                      f"PDE Loss: {avg_pde_loss:.6f}, "
                      f"Boundary Loss: {avg_boundary_loss:.6f}, "
                      f"Data Loss: {avg_data_loss:.6f}")

                # Log network weights histograms every 500 epochs
                for name, param in self.model.named_parameters():
                    self.writer.add_histogram(f'Parameters/{name}', 
                                           param.data.cpu().numpy(), 
                                           epoch)

            # Pass average loss to scheduler
            self.scheduler.step(avg_loss)
        
        # Close TensorBoard writer
        self.writer.close()

    def save_model(self, checkpoint_name="checkpoint.pth"):
        """
        Save the model and optimizer states.
        
        Parameters:
            checkpoint_name (str): The name of the checkpoint file to save.
        """
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config
        }
        checkpoint_path = os.path.join(self.checkpoint_dir, checkpoint_name)
        torch.save(checkpoint, checkpoint_path)
        print(f"Model saved to {checkpoint_path}")
    
    def evaluate(self):
        """Evaluate PINN solution for the specified model."""
        self.model.eval()
        with torch.no_grad():
            # Generate grid points based on model type
            if self.model_type == "Heston":
                S, t, v = generate_grid_points(
                    K=self.config["model"]["K"],
                    T=self.config["model"]["T"],
                    num_points=self.config["training"]["num_points"],
                    device=self.device,
                    model_type="Heston",
                    uniform=self.config["training"]["uniform"],
                )
                V_pinn = self.model(S, t, v).cpu().numpy()
            else:  # BlackScholes
                S, t = generate_grid_points(
                    K=self.config["model"]["K"], 
                    T=self.config["model"]["T"],
                    num_points=self.config["training"]["num_points"],
                    device=self.device,
                    model_type="BlackScholes",
                    uniform=self.config["training"]["uniform"],
                )
                V_pinn = self.model(S, t).cpu().numpy()

            # Compute theoretical solution based on model type
            V_theoretical = np.zeros_like(V_pinn)
            for i in range(len(S)):
                if self.model_type == "Heston":
                    V_theoretical[i] = HestonSolver.compute_solution(
                        S[i].item(), 
                        t[i].item(),
                        v[i].item(),
                        self.config
                    )
                else:  # BlackScholes
                    V_theoretical[i] = BlackScholesSolver.compute_solution(
                        S[i].item(),
                        t[i].item(),
                        self.config
                    )

            # Plot results and compute metrics
            plot_comparison(S.cpu().numpy(), t.cpu().numpy(), V_pinn, V_theoretical, self.plot_dir)

            mse = np.mean((V_pinn - V_theoretical) ** 2)
            mae = np.mean(np.abs(V_pinn - V_theoretical))
            print(f"Mean Squared Error: {mse:.6f}")
            print(f"Mean Absolute Error: {mae:.6f}")
            
            # Log metrics
            self.writer.add_scalar('Evaluation/MSE', mse)
            self.writer.add_scalar('Evaluation/MAE', mae)

