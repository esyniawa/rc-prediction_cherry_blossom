import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


class Reservoir(nn.Module):
    def __init__(self,
                 dim_reservoir=1000,
                 dim_input=1,
                 dim_output=1,
                 tau=10.0,
                 g=1.5,
                 p=0.1,
                 sigma=1.0,
                 feedback_scaling=1.0,
                 device='cuda' if torch.cuda.is_available() else 'cpu'):
        super().__init__()

        # Model parameters
        self.dim_reservoir = dim_reservoir
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.tau = tau
        self.g = g
        self.p = p
        self.sigma = sigma
        self.device = device

        # Move model to specified device
        self.to(device)

        # Initialize weights
        self.W = self._initialize_reservoir_weights()
        self.W_in = (torch.randn(dim_reservoir, dim_input) / np.sqrt(dim_input)).to(device)
        self.W_fb = (torch.randn(dim_reservoir, dim_output) *
                     feedback_scaling / np.sqrt(dim_output)).to(device)
        self.W_out = torch.zeros(dim_output, dim_reservoir).to(device)

        # Initialize states
        self.reset_state()

    def _initialize_reservoir_weights(self):
        # Create mask for sparse connectivity
        mask = (torch.rand(self.dim_reservoir, self.dim_reservoir) < self.p).float().to(self.device)
        random_weights = torch.randn(self.dim_reservoir, self.dim_reservoir).to(self.device)
        W = (random_weights * mask) * (self.g / np.sqrt(self.p * self.dim_reservoir))
        return W

    def forward(self, input_signal, dt=0.1):
        # Ensure input is on correct device and properly shaped
        input_signal = input_signal.to(self.device).view(self.dim_input)

        # Compute total input to reservoir neurons
        total_input = (torch.matmul(self.W, self.r) +
                       torch.matmul(self.W_in, input_signal.float()) +
                       torch.matmul(self.W_fb, self.output.float()))

        # Update reservoir state
        dr = (-self.r + torch.tanh(total_input)) / self.tau
        self.r = self.r + dt * dr

        # Compute output
        self.output = torch.matmul(self.W_out, self.r)

        return self.output

    def reset_state(self):
        self.r = torch.zeros(self.dim_reservoir).to(self.device)
        self.output = torch.zeros(self.dim_output).to(self.device)


class ForceTrainer:
    def __init__(self, reservoir, alpha=1.0):
        self.reservoir = reservoir
        self.alpha = alpha
        self.P = torch.eye(reservoir.dim_reservoir).to(reservoir.device) / alpha

    def train_step(self, input_signal, target, dt=0.1):
        # Ensure target is on correct device
        target = target.to(self.reservoir.device)

        # Run reservoir forward
        output = self.reservoir.forward(input_signal, dt)

        # Compute error
        error = output - target

        # Update P matrix
        r = self.reservoir.r
        Pr = torch.matmul(self.P, r)
        rPr = torch.matmul(r, Pr)
        c = 1.0 / (1.0 + rPr)
        self.P = self.P - torch.outer(Pr, Pr) * c

        # Update output weights
        self.reservoir.W_out = self.reservoir.W_out - error.view(-1, 1) @ (c * Pr).view(1, -1)

        return error


def train_and_test_sine_to_cosine():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Parameters
    reservoir_size = 1000
    simulation_time = 10_000  # time steps
    dt = .1
    frequency = 10.0  # Hz

    # Create time points and move to device
    time = torch.arange(0, simulation_time * dt, dt)

    # Create input (sine) and target (cosine) signals
    input_signal = torch.sin(2 * np.pi * frequency * time).to(device)
    target_signal = torch.cos(2 * np.pi * frequency * time).to(device)

    # Initialize reservoir
    reservoir = Reservoir(
        dim_reservoir=reservoir_size,
        dim_input=1,
        dim_output=1,
        tau=10.0,
        g=1.5,
        p=0.1,
        device=device
    )

    # Initialize trainer
    trainer = ForceTrainer(reservoir, alpha=1.0)

    # Storage for results (keep on CPU for plotting)
    outputs = torch.zeros_like(time)
    errors = torch.zeros_like(time)

    # Training phase
    print("Starting training...")
    with torch.no_grad():  # No need to track gradients for this training
        for t in range(len(time)):
            error = trainer.train_step(input_signal[t], target_signal[t], dt)
            outputs[t] = reservoir.output.cpu()  # Move to CPU for storage
            errors[t] = error.cpu()  # Move to CPU for storage

            if t % 100 == 0:
                print(f"Step {t}, Error: {error.item():.4f}")

    # Move signals back to CPU for plotting
    input_signal = input_signal.cpu()
    target_signal = target_signal.cpu()

    # Plotting results
    plt.figure(figsize=(15, 10))

    # Plot input, target, and output signals
    plt.subplot(2, 1, 1)
    plt.plot(time.numpy(), input_signal.numpy(), label='Input (sine)', alpha=0.7)
    plt.plot(time.numpy(), target_signal.numpy(), label='Target (cosine)', alpha=0.7)
    plt.plot(time.numpy(), outputs.numpy(), label='Output', alpha=0.7)
    plt.xlabel('Time')
    plt.ylabel('Signal')
    plt.title('Sine to Cosine Conversion')
    plt.legend()
    plt.grid(True)

    # Plot error
    plt.subplot(2, 1, 2)
    plt.plot(time.numpy(), errors.numpy(), label='Error')
    plt.xlabel('Time')
    plt.ylabel('Error')
    plt.title('Training Error Over Time')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    train_and_test_sine_to_cosine()
