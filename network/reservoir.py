import numpy as np
import networkx as nx

from typing import Optional


def generate_adjacency_matrix(dim_reservoir: int, rho: float, sigma: float, weights: str = 'normal'):
    """
    Generates a sparse adjacency matrix based on the Erdős–Rényi model

    :param dim_reservoir: The number of nodes
    :param rho: Scaling of recurrent nodes
    :param sigma: Probability for edge creation
    :param weights: Distribution for weight initialization. If not normal, then it is uniform
    :return: Weight matrix
    """

    graph = nx.gnp_random_graph(dim_reservoir, sigma)
    graph = nx.to_numpy_array(graph)

    # Ensure random_array is of the same shape as the graph adjacency matrix.
    if weights == 'normal':
        random_array = np.random.normal(loc=0, scale=np.sqrt(1/(sigma * dim_reservoir)), size=(dim_reservoir, dim_reservoir))
    # else Uniform
    else:
        random_array = np.random.uniform(-1, 1, size=(dim_reservoir, dim_reservoir))

    return rho * graph * random_array


def scale_matrix(W_rec: np.ndarray, rho: float):
    """
    :param W_rec: Adjacency matrix
    :param rho: Scaling of recurrent nodes
    :return:
    """
    eigenvalues, _ = np.linalg.eig(W_rec)
    max_eigenvalue = np.amax(eigenvalues)
    W_rec = W_rec / np.absolute(max_eigenvalue) * rho
    return W_rec


class RCNetwork:
    def __init__(self,
                 dim_reservoir: int,
                 dim_in: int,
                 dim_out: int,
                 alpha: float = 0.2,
                 rho: float = 1.2,
                 sigma_rec: float = 0.1,
                 sigma_in: float = 1.0,
                 feedback_connection: bool = True):

        """
        Initializes a Reservoir Computing (RC) network.

        Args:
            dim_reservoir (int): The number of nodes in the reservoir.
            dim_in (int): The number of input nodes.
            dim_out (int): The number of output nodes.
            alpha (float, optional): The learning rate for the RC network. Defaults to 0.2.
            rho (float, optional): The scaling or choas factor for recurrent nodes. Defaults to 1.2.
            sigma_rec (float, optional): The probability for edge creation in the reservoir. Defaults to 0.1.
            sigma_in (float, optional): The scaling factor for input weights. Defaults to 1.0.
            feedback_connection (bool, optional): Whether to include feedback connections from output to reservoir. Defaults to True.

        Initializes the RC network with the specified dimensions and weights for the reservoir, input, and output layers.
        The weights are initialized using the `generate_adjacency_matrix` function.

        If `feedback_connection` is True, also initializes feedback weights using random uniform distribution.

        Initializes the "firing rates" for the reservoir, input, and output nodes.

        Initializes the error vectors for force learning.

        Initializes the matrix P for the Recursive Least Squares (RLS) algorithm with the specified alpha value.

        Returns:
            None
        """

        # parameters
        self.dim_reservoir = dim_reservoir
        self.fb = feedback_connection

        # initialize weights
        self.W_rec = generate_adjacency_matrix(dim_reservoir, rho, sigma_rec)
        self.W_in = sigma_in * np.random.uniform(-1, 1, size=(dim_reservoir, dim_in))
        self.W_out = np.zeros((dim_out, dim_reservoir))

        if self.fb:
            self.W_fb = np.random.uniform(-1, 1, size=(dim_reservoir, dim_out))

        # "firing rates"
        self.x_reservoir = np.zeros(dim_reservoir)
        self.r_reservoir = np.zeros(dim_reservoir)
        self.r_in = np.zeros(dim_in)
        self.r_out = np.zeros(dim_out)

        # Force learning
        self.error_minus = np.zeros(dim_out)
        self.error_plus = np.zeros(dim_out)

        self.P = np.eye(dim_reservoir) / alpha

    def step(self, target: np.ndarray, dt: float = 1., tau: float = 10., train: bool = True):

        """
        Performs a given time step of the RCNetwork.
        Args:
            target (np.ndarray): The target output for the network.
            dt (float, optional): The time step size. Defaults to 1.
            tau (float, optional): The time constant for the RCNetwork. Defaults to 10.
            train (bool, optional): Whether to train the network. Defaults to True.
        Returns:
            None
        Description:
            This function performs a single time step of the RCNetwork. It calculates the derivative of the reservoir
            state based on the current reservoir state, input weights, and recurrent weights. It then updates the
            reservoir state and output based on the calculated derivative and the time step size. If the `train` flag is
            set to True, it also calculates the error between the current output and the target output, and updates
            the weights using the `rls` method.
        Note:
            If the `feedback_connection` flag is set to True, the function also includes the output weights in the
            calculation of the derivative.
        """

        if self.fb:
            dxdt = (-self.x_reservoir + np.dot(self.W_rec, self.r_reservoir) +
                    np.dot(self.W_in, self.r_in) + np.dot(self.W_fb, self.r_out)) / tau
        else:
            dxdt = (-self.x_reservoir + np.dot(self.W_rec, self.r_reservoir) + np.dot(self.W_in, self.r_in)) / tau

        self.x_reservoir += dxdt * dt
        self.r_reservoir = np.tanh(self.x_reservoir)
        self.r_out = np.dot(self.W_out, self.r_reservoir)

        if train:
            self.error_minus = self.r_out - target
            self._rls()
            self.error_plus = np.dot(self.W_out, self.r_reservoir) - target

    def reset_state(self):
        """
        Resets the state of the reservoir to zero.

        This function sets the state variables `x_reservoir`, `r_reservoir`, `r_in`, and `r_out` to zero.
        """

        self.x_reservoir = np.zeros(self.x_reservoir.shape)
        self.r_reservoir = np.zeros(self.r_reservoir.shape)
        self.r_in = np.zeros(self.r_in.shape)
        self.r_out = np.zeros(self.r_out.shape)

    def _rls(self):
        """
        Update the weights of the reservoir using the Recursive Least Squares (RLS) algorithm.

        This function calculates the projection matrix Pr, the scalar c, and the delta weight matrix dw.
        It then updates the weights of the reservoir by subtracting dw from W_out.

        """
        Pr = np.dot(self.P, self.r_reservoir)
        rPr = np.dot(self.r_reservoir.T, Pr)
        c = float(1.0 / (1.0 + rPr))
        self.P -= c * np.outer(Pr, Pr)

        dw = -c * np.outer(self.error_minus, Pr)
        self.W_out += dw

    def run(self,
            data_target: np.ndarray,
            data_in: np.ndarray,
            rls_training: bool = True,
            do_reset: bool = True,
            record_error: bool = False):
        """
        Run the reservoir network on the given input and target data.

        Args:
            data_target (np.ndarray): The target data of shape (t, dim_system), where t is the number of timesteps.
            data_in (np.ndarray): The reservoir input data of shape (t, dim_system).
            rls_training (bool, optional): Whether to perform RLS training. Defaults to True.
            do_reset (bool, optional): Whether to reset the reservoir activity after each iteration. Defaults to True.
            record_error (bool, optional): Whether to record the training error. Defaults to False.

        Returns:
            np.ndarray or tuple: If `record_error` is False, returns the output of the reservoir network of shape (t, dim_system).
            If `record_error` is True, returns a tuple containing the output of the reservoir network and the training error.
            The training error is a list of tuples, where each tuple contains the error_minus and error_plus values for each iteration.
        """

        # recordings
        z = np.zeros((data_target.shape[0], self.r_out.shape[0]))
        if record_error:
            training_error = []

        for i in range(data_target.shape[0]):

            # simulate
            self.r_in = data_in[i]
            self.step(target=data_target[i], train=rls_training)
            z[i] = self.r_out

            if record_error:
                training_error.append((self.error_minus, self.error_plus))

        if do_reset:
            self.reset_state()

        if record_error:
            return z, training_error
        else:
            return z

    def animate_training(self,
                         data_in: np.ndarray,
                         data_target: np.ndarray,
                         save_name: str | None):

        from .utils import find_largest_factors

        pass

    def predict(self,
                data_target: np.ndarray,
                data_in: np.ndarray):

        prediction = self.run(data_target=data_target, data_in=data_in, rls_training=False)

        return prediction

    @staticmethod
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

            a1 = np.random.RandomState(seed + out).normal(loc=0, scale=1)
            a2 = np.random.RandomState(seed + out).normal(loc=0, scale=1)
            a3 = np.random.RandomState(seed + out).normal(loc=0, scale=0.5)

            y[:, out] = a1 * np.sin(2 * np.pi * x / T) + a2 * np.sin(4 * np.pi * x / T) + a3 * np.sin(6 * np.pi * x / T)

        return y, T

    def train_dynamic_target(self,
                             n_period_train: int,
                             n_period_test: int,
                             do_plot: bool = False,
                             seed: Optional[int] = None):

        data_target, period = RCNetwork.make_dynamic_target(self.r_out.shape[0], n_period_train + n_period_test,
                                                            seed=seed)

        z_train, error = self.run(data_target=data_target[:int(n_period_train*period)],
                                  data_in=data_target[:int(n_period_train*period)],
                                  rls_training=True,
                                  record_error=True)

        true_target = data_target[int(n_period_train*period):]
        z_prediction = self.run(data_target=true_target, data_in=true_target, rls_training=False)

        mse = ((true_target - z_prediction) ** 2).mean()

        print(f'MSE = {mse:.4f}')
        if do_plot:
            import matplotlib.pyplot as plt

            error = np.array(error)

            z = np.concatenate((z_train, z_prediction))
            fig, axs = plt.subplots(nrows=2)
            axs[0].plot(data_target, c='b')
            axs[0].plot(z, c='r', marker='.',
                        linestyle='None', alpha=0.4)
            axs[0].set_title('Target (blue) | Output reservoir (red)')

            axs[1].plot(error[:, 1, :] - error[:, 0, :])
            axs[1].set_title('error_plus - error_minus')
            plt.show()
            plt.close(fig)
