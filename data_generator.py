import numpy as np

class StochasticModel:
    """Base class for stochastic models."""

    def __init__(self, S0, T, num_paths, num_steps):
        """
        Initialize the stochastic model.

        Parameters:
            S0 (float or np.ndarray): Initial stock price(s).
            T (float): Time to maturity.
            num_paths (int): Number of paths to simulate.
            num_steps (int): Number of time steps per path.
        """
        self.S0 = S0
        self.T = T
        self.num_paths = num_paths
        self.num_steps = num_steps
        self.dt = T / num_steps  # Time step size
        self.t = np.linspace(0, T, num_steps)  # Time grid

    def generate_paths(self):
        """Method to be implemented in subclasses."""
        raise NotImplementedError("Subclasses must implement `generate_paths`.")


class GeometricBrownianMotion(StochasticModel):
    """Simulate paths using Geometric Brownian Motion."""

    def __init__(self, S0, mu, sigma, T, num_paths, num_steps):
        """
        Initialize the GBM model.

        Parameters:
            S0 (float): Initial stock price.
            mu (float): Drift (expected return).
            sigma (float): Volatility.
            T (float): Time to maturity.
            num_paths (int): Number of paths to simulate.
            num_steps (int): Number of time steps per path.
        """
        super().__init__(S0, T, num_paths, num_steps)
        self.mu = mu
        self.sigma = sigma

    def generate_paths(self):
        """Generate paths for GBM."""
        # Generate random increments for Brownian motion
        dW = np.random.normal(0, np.sqrt(self.dt), size=(self.num_paths, self.num_steps - 1))

        # Initialize paths
        paths = np.zeros((self.num_paths, self.num_steps))
        paths[:, 0] = self.S0

        # Vectorized simulation of GBM
        paths[:, 1:] = self.S0 * np.exp(
            np.cumsum((self.mu - 0.5 * self.sigma**2) * self.dt + self.sigma * dW, axis=1)
        )

        return self.t, paths


class CorrelatedSDE(StochasticModel):
    """Simulate correlated asset price paths."""

    def __init__(self, S0, mu, sigma, T, num_paths, num_steps, correlation_matrix):
        """
        Initialize the Correlated SDE model.

        Parameters:
            S0 (np.ndarray): Initial stock prices for multiple assets.
            mu (np.ndarray): Drifts (expected returns) for multiple assets.
            sigma (np.ndarray): Volatilities for multiple assets.
            T (float): Time to maturity.
            num_paths (int): Number of paths to simulate.
            num_steps (int): Number of time steps per path.
            correlation_matrix (np.ndarray): Correlation matrix for assets.
        """
        super().__init__(S0, T, num_paths, num_steps)
        self.mu = mu
        self.sigma = sigma
        self.correlation_matrix = correlation_matrix

    def generate_paths(self):
        """Generate correlated paths for multiple assets."""
        num_assets = len(self.S0)
        chol_matrix = np.linalg.cholesky(self.correlation_matrix)

        # Generate independent Brownian motions
        dW_independent = np.random.normal(0, np.sqrt(self.dt), size=(num_assets, self.num_paths, self.num_steps - 1))

        # Correlate Brownian motions
        dW_correlated = np.einsum('ij,jkp->ikp', chol_matrix, dW_independent)

        # Initialize paths
        paths = np.zeros((num_assets, self.num_paths, self.num_steps))
        paths[:, :, 0] = self.S0[:, None]  # Initial prices for all assets

        # Vectorized simulation of correlated SDEs
        for i in range(num_assets):
            paths[i, :, 1:] = self.S0[i] * np.exp(
                np.cumsum((self.mu[i] - 0.5 * self.sigma[i]**2) * self.dt + self.sigma[i] * dW_correlated[i], axis=1)
            )

        return self.t, paths


class HestonModel(StochasticModel):
    """Simulate paths using the Heston Stochastic Volatility Model."""

    def __init__(self, S0, v0, kappa, theta, sigma_v, rho, r, T, num_paths, num_steps):
        """
        Initialize the Heston model.

        Parameters:
            S0 (float): Initial stock price.
            v0 (float): Initial variance.
            kappa (float): Mean reversion rate of variance.
            theta (float): Long-term variance.
            sigma_v (float): Volatility of variance.
            rho (float): Correlation between stock price and variance.
            r (float): Risk-free rate.
            T (float): Time to maturity.
            num_paths (int): Number of paths to simulate.
            num_steps (int): Number of time steps per path.
        """
        super().__init__(S0, T, num_paths, num_steps)
        self.v0 = v0
        self.kappa = kappa
        self.theta = theta
        self.sigma_v = sigma_v
        self.rho = rho
        self.r = r

    def generate_paths(self):
        """Generate stock price and variance paths for the Heston model."""
        dt = self.dt

        # Initialize arrays
        S = np.zeros((self.num_paths, self.num_steps))
        v = np.zeros((self.num_paths, self.num_steps))
        S[:, 0] = self.S0
        v[:, 0] = self.v0

        # Correlated Brownian motions
        Z1 = np.random.normal(size=(self.num_paths, self.num_steps - 1))
        Z2 = np.random.normal(size=(self.num_paths, self.num_steps - 1))
        dW_S = Z1
        dW_v = self.rho * Z1 + np.sqrt(1 - self.rho**2) * Z2

        # Vectorized updates
        v[:, 1:] = np.maximum(
            v[:, :-1] + self.kappa * (self.theta - v[:, :-1]) * dt +
            self.sigma_v * np.sqrt(np.maximum(v[:, :-1], 0) * dt) * dW_v,
            0  # Variance must remain non-negative
        )

        S[:, 1:] = S[:, :-1] * np.exp(
            (self.r - 0.5 * v[:, :-1]) * dt + np.sqrt(np.maximum(v[:, :-1], 0) * dt) * dW_S
        )

        return self.t, S, v



class JumpDiffusion(StochasticModel):
    """Simulate paths using Merton's Jump Diffusion model."""

    def __init__(self, S0, mu, sigma, T, num_paths, num_steps, lambda_jump, mu_jump, sigma_jump):
        """
        Initialize the Jump Diffusion model.

        Parameters:
            S0 (float): Initial stock price.
            mu (float): Drift (expected return).
            sigma (float): Volatility.
            T (float): Time to maturity.
            num_paths (int): Number of paths to simulate.
            num_steps (int): Number of time steps per path.
            lambda_jump (float): Jump intensity (average number of jumps per year).
            mu_jump (float): Mean of log jump size.
            sigma_jump (float): Std deviation of log jump size.
        """
        super().__init__(S0, T, num_paths, num_steps)
        self.mu = mu
        self.sigma = sigma
        self.lambda_jump = lambda_jump
        self.mu_jump = mu_jump
        self.sigma_jump = sigma_jump

    def generate_paths(self):
        """Generate paths for Jump Diffusion."""
        dW = np.random.normal(0, np.sqrt(self.dt), size=(self.num_paths, self.num_steps - 1))

        # Simulate Poisson jumps
        jump_counts = np.random.poisson(self.lambda_jump * self.dt, size=(self.num_paths, self.num_steps - 1))
        jump_sizes = np.random.normal(self.mu_jump, self.sigma_jump, size=jump_counts.shape)

        # Initialize paths
        paths = np.zeros((self.num_paths, self.num_steps))
        paths[:, 0] = self.S0

        # Vectorized simulation of Jump Diffusion
        paths[:, 1:] = self.S0 * np.exp(
            np.cumsum((self.mu - 0.5 * self.sigma**2) * self.dt + self.sigma * dW + jump_counts * jump_sizes, axis=1)
        )

        return self.t, paths
