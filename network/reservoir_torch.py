import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output

import os
from typing import Optional, Iterator


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
                 w_out_initialization: Optional[str] = None,  # None means zeros else 'uniform' or 'normal'
                 noise_scaling: float = 0.05,
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
        self.W_rec = self._initialize_reservoir_weights().to(device)
        self.W_in = (torch.empty(dim_reservoir, dim_input).uniform_(-1, 1) * feedforward_scaling).to(device)
        self.W_fb = (torch.empty(dim_reservoir, dim_output).uniform_(-1, 1) * feedback_scaling).to(device)
        self.feedback_scaling = feedback_scaling

        # initialize readout weights
        if w_out_initialization == 'uniform':
            self.W_out = torch.empty(dim_output, dim_reservoir).uniform_(-1, 1).to(device)
        elif w_out_initialization == 'normal':
            self.W_out = (torch.randn(dim_output, dim_reservoir) / np.sqrt(dim_input)).to(device)
        else:
            self.W_out = torch.zeros(dim_output, dim_reservoir).to(device)

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
    def forward(self, input_signal, dt: float = 0.1, compute_output: bool = True):
        # Ensure input is on correct device and properly shaped
        input_signal = input_signal.to(self.device).view(self.dim_input)

        r_tanh = torch.tanh(self.r)

        # Compute total input to reservoir neurons
        state = (torch.matmul(self.W_rec, r_tanh) +
                 torch.matmul(self.W_in, input_signal.float()) +
                 torch.randn(self.dim_reservoir).to(self.device) * self.noise_scaling)

        if self.feedback_scaling:
            state += torch.matmul(self.W_fb, self.output.float())

        # Update reservoir state
        dr = (-self.r + state)
        self.r += (dt / self.tau) * dr

        # Compute output
        if compute_output:
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
                   dt: float = 0.1,
                   w_update: bool = True,
                   ):

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

        if w_update:
            # Update P matrix
            r = self.reservoir.r
            Pr = torch.matmul(self.P, r)
            rPr = torch.matmul(r, Pr)
            c = 1.0 / (1.0 + rPr)
            self.P -= c * torch.outer(Pr, Pr)

            # Update output weights
            self.reservoir.W_out -= c * torch.outer(error_minus, Pr)

            # Error after update
            error_plus = self.reservoir.step() - target

            return error_minus, error_plus
        else:
            return error_minus, torch.empty_like(error_minus)


class FullForceTrainer:
    def __init__(self,
                 task_network: Reservoir,
                 alpha: float = 1.0,
                 seed: Optional[int] = None,
                 clone_input_weights: bool = True,  # clones input weights from task network to the target network
                 clone_target_weights: bool = False,
                 set_recurrent_weights_to_zeros: bool = True,
                 # In the implementation of the paper, they set the recurrent weights of the task network to zero,
                 # also to asure no chaotic reservoir
                 ):
        self.task_network = task_network
        self.device = task_network.device

        # Create target network
        self.target_network = Reservoir(
            dim_reservoir=task_network.dim_reservoir,
            dim_input=task_network.dim_input + task_network.dim_output,
            # Input + target output dims (both will be given to the network)
            dim_output=task_network.dim_output,
            tau=task_network.tau,
            chaos_factor=1.0,  # No chaotic reservoir due to no feedback
            probability_recurrent_connection=task_network.probability_recurrent_connection,
            feedback_scaling=0.0,  # No feedback in either network
            seed=seed,
            device=self.device
        )

        # Disable feedback
        self.task_network.W_fb.zero_()
        self.task_network.feedback_scaling = 0.0
        self.target_network.W_fb.zero_()

        # Some initialisations from the authors
        if clone_input_weights:
            self.target_network.W_in[:, :self.task_network.dim_input] = self.task_network.W_in[:, :self.task_network.dim_input]

        if clone_target_weights:
            self.target_network.W_in[:, self.task_network.dim_input:] = self.task_network.W_in[:, :self.task_network.dim_output]

        if set_recurrent_weights_to_zeros:
            self.task_network.W_rec.zero_()

        # Initialize single P matrix for RLS
        self.P = torch.eye(task_network.dim_reservoir).to(self.device) / alpha
        self.W_in_target = self.target_network.W_in[:, self.task_network.dim_input:]

    @torch.no_grad()
    def train_step(self,
                   input_signal: torch.Tensor,
                   target: torch.Tensor,
                   dt: float = 0.1,
                   w_update: bool = True,  # Maybe you don't want to update the weights in every iteration
                   ):
        # Ensure inputs are on correct device
        input_signal = input_signal.to(self.device)
        target = target.to(self.device)

        # Run networks forward
        self.target_network.forward(torch.cat([input_signal, target]), dt,
                                    compute_output=False)  # Output not needed for training
        self.task_network.forward(input_signal, dt)

        if w_update:
            # Get rates
            r = self.task_network.r
            rd = self.target_network.r

            # Compute errors
            J_err = (torch.matmul(self.task_network.W_rec, r) -
                     torch.matmul(self.target_network.W_rec, rd) -
                     torch.matmul(self.W_in_target, target))

            error_minus = torch.matmul(self.task_network.W_out, r) - target

            # Update P matrix and compute gain properly
            Pr = torch.matmul(self.P, r)
            rPr = torch.matmul(r, Pr)
            k = Pr / (1 + rPr)  # with gain
            self.P -= torch.outer(Pr, k)

            # Update weights
            self.task_network.W_rec -= torch.outer(J_err, k)
            self.task_network.W_out -= torch.outer(error_minus, k)

            # Error after update
            error_plus = self.task_network.step() - target

            return error_minus, error_plus

        else:
            error_minus = torch.matmul(self.task_network.W_out, torch.tanh(self.task_network.r)) - target
            return error_minus, torch.empty_like(error_minus)

    def reset_states(self):
        self.task_network.reset_state()
        self.target_network.reset_state()


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

        a1 = np.random.RandomState(seed + out).normal(loc=0, scale=2)
        a2 = np.random.RandomState(seed + out).normal(loc=0, scale=2)
        a3 = np.random.RandomState(seed + out).normal(loc=0, scale=1)

        y[:, out] = a1 * np.sin(2 * np.pi * x / T) + a2 * np.sin(4 * np.pi * x / T) + a3 * np.sin(6 * np.pi * x / T)

    y /= np.amax(y)

    return y, T


def test_reservoir(seed: Optional[int] = None,
                   full_force: bool = False,):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Parameters
    reservoir_size = 500
    dt = 0.001
    dim_input = 3
    target_signal, period = make_dynamic_target(dim_input, n_periods=100, seed=seed)

    # Convert target signal to torch tensor
    target_signal = torch.from_numpy(target_signal).float()

    # Initialize reservoir
    reservoir = Reservoir(
        dim_reservoir=reservoir_size,
        dim_input=dim_input,
        dim_output=dim_input,
        tau=0.01,
        chaos_factor=1.5,
        probability_recurrent_connection=1.0,
        device=device,
        feedback_scaling=1.0,
        w_out_initialization=None,
    )

    # Initialize trainer
    if full_force:
        trainer = FullForceTrainer(reservoir, alpha=1.0)
    else:
        trainer = ForceTrainer(reservoir, alpha=1.0)

    # Training parameters
    n_steps = len(target_signal)
    plot_interval = 500  # Update plot every 100 steps
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
    ax1.set_ylim(-1.2, 1.2)
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
            error_minus, _ = trainer.train_step(
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
    reservoir, errors = test_reservoir(seed=seed, full_force=True)
