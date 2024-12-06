import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output

import os
from typing import Optional, Iterator

from torch.nn import Parameter


class Reservoir(nn.Module):
    def __init__(self,
                 dim_reservoir: int,
                 dim_input: int,
                 dim_output: int,
                 tau: float = 10.0,
                 chaos_factor: float = 1.5,
                 probability_recurrent_connection: float = 0.1,
                 feedforward_scaling: float = 1.0,
                 feedback_scaling: float = 1.0,
                 noise_scaling: float = 0.0,
                 seed: Optional[int] = None,
                 device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        super().__init__()
        # Set random seed
        if seed is not None:
            torch.manual_seed(seed)

        # Model parameters
        self.dim_reservoir = dim_reservoir
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.tau = tau
        self.chaos_factor = chaos_factor
        self.probability_recurrent_connection = probability_recurrent_connection
        self.device = device

        # Move model to specified device
        self.to(device)

        # Initialize weights
        self.W = self._initialize_reservoir_weights().to(device)
        self.W_in = (torch.randn(dim_reservoir, dim_input) * feedforward_scaling / np.sqrt(dim_input)).to(device)
        self.W_fb = (torch.randn(dim_reservoir, dim_output) * feedback_scaling / np.sqrt(dim_output)).to(device)
        self.W_out = torch.zeros(dim_output, dim_reservoir).to(device)  # initialize readout weights

        # Initialize states
        self.noise_scaling = noise_scaling
        self.reset_state()

    def _initialize_reservoir_weights(self):
        """
        Initializes the reservoir weight matrix using sparse connectivity and spectral scaling

        Returns:
            torch.Tensor: Initialized weight matrix scaled to desired spectral radius
        """
        # Create sparse mask using bernoulli distribution
        mask = (torch.rand(self.dim_reservoir, self.dim_reservoir) < self.probability_recurrent_connection).float()

        # Initialize weights using normal distribution
        weights = torch.randn(self.dim_reservoir, self.dim_reservoir) * \
                  torch.sqrt(torch.tensor(1.0 / (self.probability_recurrent_connection * self.dim_reservoir)))

        # Apply mask to create sparse connectivity
        W = mask * weights

        # Scale matrix to desired spectral radius (chaos_factor)
        eigenvalues = torch.linalg.eigvals(W)
        max_abs_eigenvalue = torch.max(torch.abs(eigenvalues))
        W = (W / max_abs_eigenvalue) * self.chaos_factor

        return W

    @torch.no_grad()
    def forward(self, input_signal, dt: float = 0.1):
        # Ensure input is on correct device and properly shaped
        input_signal = input_signal.to(self.device).view(self.dim_input)

        # Compute total input to reservoir neurons
        state = (torch.matmul(self.W, self.r) +
                 torch.matmul(self.W_in, input_signal.float()) +
                 torch.matmul(self.W_fb, self.output.float()) +
                 torch.randn(self.dim_reservoir).to(self.device) * self.noise_scaling)

        # Update reservoir state
        dr = (-self.r + torch.tanh(state)) / self.tau
        self.r += dt * dr

        # Compute output
        self.output = self.step()

        return self.output

    @torch.no_grad()
    def step(self):
        return torch.matmul(self.W_out, self.r)

    def reset_state(self):
        self.r = torch.zeros(self.dim_reservoir).to(self.device)
        self.output = torch.zeros(self.dim_output).to(self.device)

    def save(self, path):
        folder = os.path.split(path)[0]
        if not os.path.exists(folder):
            os.makedirs(folder)

        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))

    def reservoir_parameters(self) -> dict:
        return {
            'dim_reservoir': self.dim_reservoir,
            'dim_input': self.dim_input,
            'dim_output': self.dim_output,
            'tau': self.tau,
            'chaos_factor': self.chaos_factor,
            'probability_recurrent_connection': self.probability_recurrent_connection,
            'device': self.device,
            'noise_scaling': self.noise_scaling
        }


class ForceTrainer:
    def __init__(self,
                 reservoir: Reservoir,
                 alpha: float = 1.0):
        self.reservoir = reservoir
        self.P = torch.eye(reservoir.dim_reservoir).to(reservoir.device) / alpha

    @torch.no_grad()
    def train_step(self,
                   input_signal: torch.Tensor | np.ndarray,
                   target: torch.Tensor | np.ndarray,
                   dt: float = 0.1):

        if isinstance(input_signal, np.ndarray):
            input_signal = torch.from_numpy(input_signal)
        if isinstance(target, np.ndarray):
            target = torch.from_numpy(target)

        # Ensure input and target are on correct device
        input_signal = input_signal.to(self.reservoir.device)
        target = target.to(self.reservoir.device)

        # Run reservoir forward
        output = self.reservoir.forward(input_signal, dt)

        # Compute error
        error_minus = output - target

        # Update P matrix
        r = self.reservoir.r
        Pr = torch.matmul(self.P, r)
        rPr = torch.matmul(r, Pr)
        c = 1.0 / (1.0 + rPr)
        self.P -= c * torch.outer(Pr, Pr)

        # Update output weights
        self.reservoir.W_out -= c * torch.outer(error_minus, Pr)

        # Error after update
        error_plus = self.reservoir.step()

        return error_minus, error_plus


def make_dynamic_target(dim_out: int, n_periods: int, seed: Optional[int] = None):
    """
    Generates a dynamic target signal for the reservoir computing network.

    :param dim_out: The dimensionality of the output signal.
    :param n_periods: The number of trials for which the signal is generated.
    :param seed: The seed for the random number generator. Default is None.

    :return: A tuple containing the generated dynamic target signal (numpy array) and the period time (float).
    """

    # random period time
    T = np.random.RandomState(seed).randint(100, 200)
    x = np.arange(0, n_periods * T)

    y = np.zeros((len(x), dim_out))

    for out in range(dim_out):
        if seed is None:
            seed = np.random.randint(0, 1000)

        a1 = np.random.RandomState(seed + out).normal(loc=0, scale=1)
        a2 = np.random.RandomState(seed + out).normal(loc=0, scale=1)
        a3 = np.random.RandomState(seed + out).normal(loc=0, scale=0.5)

        y[:, out] = a1 * np.sin(2 * np.pi * x / T) + a2 * np.sin(4 * np.pi * x / T) + a3 * np.sin(6 * np.pi * x / T)

    return y, T


def test_reservoir(seed: Optional[int] = None):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Parameters
    reservoir_size = 500
    dt = 0.1
    dim_input = 3
    target_signal, period = make_dynamic_target(dim_input, n_periods=10, seed=seed)

    # Convert target signal to torch tensor
    target_signal = torch.from_numpy(target_signal).float()

    # Initialize reservoir
    reservoir = Reservoir(
        dim_reservoir=reservoir_size,
        dim_input=dim_input,
        dim_output=dim_input,
        tau=10.0,
        chaos_factor=1.5,
        probability_recurrent_connection=0.1,
        device=device
    )

    # Initialize trainer
    trainer = ForceTrainer(reservoir, alpha=0.2)

    # Training parameters
    n_steps = len(target_signal)
    plot_interval = 100  # Update plot every 100 steps
    error_history = []
    output_history = []

    # Create figure for live plotting
    plt.ion()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    lines_target = []
    lines_output = []

    # Initialize plot lines
    for i in range(dim_input):
        target_line, = ax1.plot([], [], '--', label=f'Target {i}')
        output_line, = ax1.plot([], [], label=f'Output {i}')
        lines_target.append(target_line)
        lines_output.append(output_line)

    ax1.set_xlabel('Time steps')
    ax1.set_ylabel('Signal')
    ax1.set_title('Target vs Output')
    ax1.legend()

    error_line, = ax2.plot([], [], label='Mean squared error')
    ax2.set_xlabel('Time steps')
    ax2.set_ylabel('MSE')
    ax2.set_title('Training Error')
    ax2.set_yscale('log')
    ax2.legend()

    # Set initial y-limits for error plot (positive log scale)
    ax2.set_ylim(1e-6, 1)

    # Training loop
    print("Starting training...")
    running_window = 20  # Window size for error calculation

    try:
        for step in range(n_steps):
            # Generate input (in this case, same as target)
            current_input = target_signal[step]
            current_target = target_signal[step]

            # Training step
            error_minus, error_plus = trainer.train_step(
                current_input,
                current_target,
                dt=dt
            )

            # Store output
            output_history.append(reservoir.output.cpu().detach().numpy())

            # Calculate error (ensure it's positive for log scale)
            mse = max(1e-10, torch.mean(error_minus ** 2).item())  # Set minimum value to avoid log(0)
            error_history.append(mse)

            # Update plot periodically
            if (step + 1) % plot_interval == 0:
                # Convert output history to numpy array for plotting
                outputs = np.array(output_history)

                # Update time window for plotting
                time_indices = np.arange(len(output_history))

                # Update signals plot
                for i in range(dim_input):
                    target_data = target_signal[:len(time_indices), i].cpu().numpy()
                    output_data = outputs[:, i]

                    lines_target[i].set_data(time_indices, target_data)
                    lines_output[i].set_data(time_indices, output_data)

                # Update error plot
                error_line.set_data(time_indices, error_history)

                # Adjust plot limits
                ax1.set_xlim(max(0, len(output_history) - running_window), len(output_history))
                ax1.set_ylim(min(target_signal.cpu().numpy().min(), outputs.min()) - 0.1,
                             max(target_signal.cpu().numpy().max(), outputs.max()) + 0.1)

                # Update error plot limits (ensure they're positive for log scale)
                if len(error_history) > 0:
                    min_error = max(1e-10, min(error_history))
                    max_error = max(error_history)
                    ax2.set_ylim(min_error / 10, max_error * 10)  # Add some padding in log space

                ax2.set_xlim(0, len(error_history))

                # Draw updated plots
                fig.canvas.draw()
                fig.canvas.flush_events()

                # Print progress
                print(f"Step {step + 1}/{n_steps}, MSE: {mse:.6e}")

    except KeyboardInterrupt:
        print("\nTraining interrupted by user")

    plt.ioff()

    # Final evaluation
    print("\nTraining completed!")
    print(f"Final MSE: {mse:.6e}")

    # Show final plot
    plt.show()

    return reservoir, error_history


if __name__ == "__main__":
    # Set random seed for reproducibility
    seed = 42
    reservoir, errors = test_reservoir(seed=seed)

