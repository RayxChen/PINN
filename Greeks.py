import torch
import matplotlib.pyplot as plt
import os
import numpy as np

class GreeksCalculator:
    """
    Module to calculate the Greeks (Delta, Gamma, Vega, Theta, Rho) for option pricing
    using Automatic Differentiation with PyTorch.
    """
    def __init__(self, PINN_solver):
        """
        Parameters:
            PINN_solver: OptionPricingSolver instance containing the trained PINN model
        """
        self.PINN_solver = PINN_solver
        self.model = PINN_solver.model
        self.config = PINN_solver.config
        self.device = PINN_solver.device
        self.model_type = PINN_solver.model_type
        
        # Create Greeks directory within PINN_solver's plot directory
        self.greeks_dir = os.path.join(PINN_solver.plot_dir, 'greeks')
        os.makedirs(self.greeks_dir, exist_ok=True)

    def calculate_delta(self, V, S):
        """Calculate Delta: ∂V/∂S"""
        return torch.autograd.grad(V, S, grad_outputs=torch.ones_like(V), create_graph=True)[0]

    def calculate_gamma(self, delta, S):
        """Calculate Gamma: ∂²V/∂S²"""
        return torch.autograd.grad(delta, S, grad_outputs=torch.ones_like(delta), create_graph=True)[0]

    def calculate_theta(self, V, t):
        """Calculate Theta: ∂V/∂t"""
        return torch.autograd.grad(V, t, grad_outputs=torch.ones_like(V), create_graph=True)[0]

    def calculate_rho(self, S, t, v, V):
        """Calculate Rho: ∂V/∂r using finite differences"""
        r = self.config["model"]["r"]
        delta_r = 1e-4
        self.config["model"]["r"] = r + delta_r
        V_r_plus = self.model(S, t, v) if self.model_type == "Heston" else self.model(S, t)
        rho = (V_r_plus - V) / delta_r
        self.config["model"]["r"] = r  # Reset risk-free rate
        return rho

    def calculate_vega(self, V, S, t, v=None):
        """Calculate Vega: ∂V/∂σ"""
        if self.model_type == "Heston":
            return torch.autograd.grad(V, v, grad_outputs=torch.ones_like(V), create_graph=True)[0]
        else:
            # For Black-Scholes, calculate Vega analytically
            sigma = self.config["model"]["sigma"]
            K = self.config["model"]["K"]
            r = self.config["model"]["r"]
            T = self.config["model"]["T"]
            
            d1 = (torch.log(S/K) + (r + 0.5*sigma**2)*(T-t)) / (sigma*torch.sqrt(T-t))
            N_prime_d1 = torch.exp(-0.5*d1**2) / torch.sqrt(2*torch.tensor(np.pi))
            return S * torch.sqrt(T-t) * N_prime_d1

    def compute_greeks(self, S, t, v=None):
        """
        Computes all Greeks for given inputs.
        """
        # Set model to evaluation mode
        self.model.eval()
        
        # Ensure inputs require gradients
        S.requires_grad = True
        t.requires_grad = True
        if v is not None:
            v.requires_grad = True

        # Forward pass: compute option price V
        V = self.model(S, t, v) if self.model_type == "Heston" else self.model(S, t)

        # Calculate all Greeks
        delta = self.calculate_delta(V, S)
        gamma = self.calculate_gamma(delta, S)
        theta = self.calculate_theta(V, t)
        rho = self.calculate_rho(S, t, v, V)
        vega = self.calculate_vega(V, S, t, v)

        # Package results
        return {
            "Delta": delta.detach(),
            "Gamma": gamma.detach(),
            "Vega": vega.detach(),
            "Theta": theta.detach(),
            "Rho": rho.detach()
        }

    def plot_greeks(self, S_range=None, t=None, v=None):
        """
        Plot the Greeks over a range of asset prices with enhanced styling.

        Parameters:
            S_range: Array of asset prices to evaluate Greeks over (optional)
            t: Time to maturity (scalar, optional)
            v: Volatility (scalar, required for Heston model, optional)
        """
        # Use default values if not provided
        if S_range is None:
            K = self.config["model"]["K"]
            S_range = np.linspace(0.5 * K, 1.5 * K, 100)
        if t is None:
            t = self.config["model"]["T"] / 2
        if v is None and self.model_type == "Heston":
            v = self.config["model"]["theta"]

        # Convert inputs to tensors
        S = torch.tensor(S_range, device=self.device, dtype=torch.float32)
        t = torch.tensor([t], device=self.device, dtype=torch.float32).expand(len(S))
        if self.model_type == "Heston":
            v = torch.tensor([v], device=self.device, dtype=torch.float32).expand(len(S))

        # Compute Greeks
        greeks = self.compute_greeks(S, t, v)

        # Set style parameters
        # plt.style.use('seaborn')
        colors = ['#2ecc71', '#e74c3c', '#3498db', '#f1c40f', '#9b59b6']
        
        # Create figure with space for all Greeks including Vega
        fig = plt.figure(figsize=(15, 12))
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

        # Add title with parameters
        K = self.config["model"]["K"]
        r = self.config["model"]["r"]
        title = f'Option Greeks vs Asset Price\nK={K}, r={r:.2f}, τ={t[0].item():.2f}'
        if self.model_type == "Heston":
            title += f', v={v[0].item():.2f}'
        fig.suptitle(title, fontsize=16, y=0.95)

        # Common plotting parameters
        plot_params = {
            'linewidth': 2.5,
            'alpha': 0.8,
        }
        
        # Function to style axes
        def style_axes(ax, title, ylabel):
            ax.set_title(title, fontsize=12, pad=10)
            ax.set_xlabel('Asset Price', fontsize=10)
            ax.set_ylabel(ylabel, fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.tick_params(labelsize=9)
            # Add vertical line at strike price
            ax.axvline(x=K, color='gray', linestyle='--', alpha=0.5, label='Strike Price')
            # Add horizontal line at y=0
            ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)

        # Plot Delta
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(S_range, greeks['Delta'].cpu(), color=colors[0], **plot_params)
        style_axes(ax1, 'Delta (∂V/∂S)', 'Delta')
        ax1.set_ylim(-0.1, 1.1)

        # Plot Gamma
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(S_range, greeks['Gamma'].cpu(), color=colors[1], **plot_params)
        style_axes(ax2, 'Gamma (∂²V/∂S²)', 'Gamma')

        # Plot Theta
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.plot(S_range, greeks['Theta'].cpu(), color=colors[2], **plot_params)
        style_axes(ax3, 'Theta (∂V/∂t)', 'Theta')

        # Plot Rho
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.plot(S_range, greeks['Rho'].cpu(), color=colors[3], **plot_params)
        style_axes(ax4, 'Rho (∂V/∂r)', 'Rho')

        # Plot Vega (now for both models)
        ax5 = fig.add_subplot(gs[2, :])
        ax5.plot(S_range, greeks['Vega'].cpu(), color=colors[4], **plot_params)
        style_axes(ax5, 'Vega (∂V/∂σ)', 'Vega')

        # Add watermark
        fig.text(0.99, 0.01, 'Generated by PINN', 
                 fontsize=8, color='gray', alpha=0.5,
                 ha='right', va='bottom')

        # Save plot with high DPI
        plt.savefig(os.path.join(self.greeks_dir, 'greeks.png'), 
                    dpi=300, bbox_inches='tight')
        plt.close()

    def black_scholes_price(self, S, K, T, t, r, sigma):
        """Calculate Black-Scholes price for given parameters"""
        tau = T - t
        d1 = (torch.log(S/K) + (r + 0.5*sigma**2)*tau) / (sigma*torch.sqrt(tau))
        d2 = d1 - sigma*torch.sqrt(tau)
        
        N_d1 = 0.5 * (1 + torch.erf(d1 / torch.sqrt(torch.tensor(2.0))))
        N_d2 = 0.5 * (1 + torch.erf(d2 / torch.sqrt(torch.tensor(2.0))))
        
        call_price = S*N_d1 - K*torch.exp(-r*tau)*N_d2
        return call_price

    def implied_volatility(self, market_price, S, t, max_iter=100, tolerance=1e-5):
        """
        Calculate implied volatility using Newton-Raphson method with better bounds and checks
        """
        K = self.config["model"]["K"]
        T = self.config["model"]["T"]
        r = self.config["model"]["r"]
        
        # Set bounds for implied volatility
        MIN_VOL = 0.01  # 1% minimum volatility
        MAX_VOL = 2.0   # 200% maximum volatility
        
        # Initial guess based on ATM volatility
        moneyness = S/K
        if 0.8 <= moneyness <= 1.2:
            sigma = torch.tensor(0.3, device=self.device)
        else:
            # Start with higher initial guess for far ITM/OTM options
            sigma = torch.tensor(0.5, device=self.device)

        # Check if price is within valid bounds
        intrinsic_value = torch.maximum(S - K, torch.tensor(0.0))
        upper_bound = S  # European call can't be worth more than underlying
        
        if market_price < intrinsic_value or market_price > upper_bound:
            return torch.tensor(float('nan'), device=self.device)

        for i in range(max_iter):
            # Calculate BS price and vega
            price = self.black_scholes_price(S, K, T, t, r, sigma)
            
            # Calculate difference
            diff = price - market_price
            
            if torch.abs(diff) < tolerance:
                break
                
            # Calculate vega
            d1 = (torch.log(S/K) + (r + 0.5*sigma**2)*(T-t)) / (sigma*torch.sqrt(T-t))
            vega = S * torch.sqrt(T-t) * torch.exp(-0.5*d1**2) / torch.sqrt(2*torch.tensor(np.pi))
            
            # Avoid division by zero
            if torch.abs(vega) < 1e-10:
                return torch.tensor(float('nan'), device=self.device)
            
            # Update sigma using Newton-Raphson
            sigma = sigma - diff / vega
            
            # Ensure sigma stays within bounds
            sigma = torch.clamp(sigma, MIN_VOL, MAX_VOL)
            
        # If no convergence or invalid result, return NaN
        if i == max_iter - 1 or torch.isnan(sigma) or torch.isinf(sigma):
            return torch.tensor(float('nan'), device=self.device)
            
        return sigma

    def plot_implied_volatility(self, S_range=None, t=None):
        """
        Plot implied volatility smile/skew with corrected moneyness scale
        """
        if S_range is None:
            K = self.config["model"]["K"]
            S_range = np.linspace(0.5 * K, 1.5 * K, 100)
        if t is None:
            t = self.config["model"]["T"] / 2

        # Convert to tensors
        S = torch.tensor(S_range, device=self.device, dtype=torch.float32)
        t = torch.tensor([t], device=self.device, dtype=torch.float32).expand(len(S))

        # Get model prices
        with torch.no_grad():
            model_prices = self.model(S, t)

        # Calculate implied volatilities
        implied_vols = []
        valid_S = []
        for i in range(len(S)):
            impl_vol = self.implied_volatility(model_prices[i], S[i], t[i])
            if not torch.isnan(impl_vol):
                implied_vols.append(impl_vol.item())
                valid_S.append(S_range[i])

        # Convert to numpy arrays for easier plotting
        valid_S = np.array(valid_S)
        implied_vols = np.array(implied_vols)

        # Create figure
        fig, ax1 = plt.subplots(figsize=(12, 8))
        
        # Plot implied volatility on primary axis
        ax1.plot(valid_S, implied_vols, 'b-', linewidth=2, label='Implied Volatility')
        
        # Add constant volatility line
        const_vol = self.config["model"]["sigma"]
        ax1.axhline(y=const_vol, color='r', linestyle='--', 
                    label=f'BS Constant Volatility ({const_vol:.2f})')

        # Add strike price line
        K = self.config["model"]["K"]
        ax1.axvline(x=K, color='gray', linestyle='--', alpha=0.5, label='Strike Price')
        
        # Primary axis styling
        ax1.set_xlabel('Spot Price', fontsize=12)
        ax1.set_ylabel('Implied Volatility', fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, min(2.0, max(implied_vols) * 1.1))
        
        # Create secondary x-axis for moneyness
        ax2 = ax1.twiny()
        
        # Set moneyness ticks
        moneyness_ticks = [0.5, 0.75, 1.0, 1.25, 1.5]
        moneyness_tick_locations = [m * K for m in moneyness_ticks]
        ax2.set_xticks(moneyness_tick_locations)
        ax2.set_xticklabels([f'{m:.2f}' for m in moneyness_ticks])
        
        # Ensure both axes have same limits
        ax2.set_xlim(ax1.get_xlim())
        
        # Secondary axis styling
        ax2.set_xlabel('Moneyness (S/K)', fontsize=12)
        
        # Title and legend
        plt.title('Implied Volatility Smile/Skew', fontsize=14)
        ax1.legend(loc='best')

        # Save plot
        plt.savefig(os.path.join(self.greeks_dir, 'implied_volatility.png'), 
                    dpi=300, bbox_inches='tight')
        plt.close()
