import numpy as np
import torch
from scipy.integrate import solve_ivp

class RegressionTasksSelkov:
    """
    Same regression task as in Finn et al. 2017 (MAML)
    """

    def __init__(self, mode="train"):
        self.num_inputs = 2
        self.num_outputs = 2

        # self.a_param = [0.1]
        # self.b_param = [b for b in np.linspace(0.1, 1.0, 10)]

        # self.input_range = [-5, 5]
        train_envs = environments = [(0.1, b) for b in list(np.linspace(-1, -0.25, 7))\
        + list(np.linspace(-0.1, 0.1, 7))\
        + list(np.linspace(0.25, 1., 7))]

        test_envs = [(0.1, b) for b in [-1.25, -0.65, -0.05, 0.02, 0.6, 1.2]]

        if mode == "train":
            self.environments = train_envs
        elif mode == "valid" or mode == "adapt_test":
            self.environments = train_envs
        elif mode == "adapt":
            self.environments = test_envs
        self.mode = mode

    def get_input_range(self, size=100):
        return torch.linspace(self.input_range[0], self.input_range[1], steps=size).unsqueeze(1)

    def sample_inputs(self, batch_size, env_id, *args, **kwargs):

        if not hasattr(self, 'data'):
            if self.mode == "train":
                self.data = np.load('regression/data_selkov/train_data.npz')
            elif self.mode == "valid":
                self.data = np.load('regression/data_selkov/test_data.npz')
            elif self.mode == "adapt":
                self.data = np.load('regression/data_selkov/adapt_data.npz')
            elif self.mode == "adapt_test":
                self.data = np.load('regression/data_selkov/adapt_test_data.npz')

        X = self.data['X']
        # inputs = X[env_id, :, :-1, :].reshape((-1, self.num_inputs))
        inputs = X[env_id, :, 0, :].reshape((-1, self.num_inputs))
        return torch.Tensor(inputs), torch.Tensor(self.data['t'])

    def sample_targets(self, batch_size, env_id, *args, **kwargs):

        if not hasattr(self, 'data'):
            if self.mode == "train":
                self.data = np.load('regression/data_selkov/train_data.npz')
            elif self.mode == "valid":
                self.data = np.load('regression/data_selkov/test_data.npz')
            elif self.mode == "adapt":
                self.data = np.load('regression/data_selkov/adapt_data.npz')
            elif self.mode == "adapt_test":
                self.data = np.load('regression/data_selkov/adapt_test_data.npz')

        X = self.data['X']
        outputs = X[env_id, :, :, :]
        # outputs = X[env_id, :, 1:, :].reshape((-1, self.num_inputs))
        return torch.Tensor(outputs).permute((1,0,2)), torch.Tensor(self.data['t'])

    def sample_task(self):
        a, b = self.environments[np.random.randint(0, len(self.environments))]
        return self.get_target_function(a, b)

    @staticmethod
    def get_target_function(a, b):
        # """ Before calling this function, remember to set the seed """
        def selkov(t, y, a, b):
            x, y = y
            dx = -x + a*y + (x**2)*y
            dy = b - a*y - (x**2)*y
            return np.array([dx, dy])
        dt = 1.

        def target_function(x):
            """ integrates the ODE from 0 to dt with initial condition x0 using solve_ivp"""
            # def solver(x0):
            #     return solve_ivp(selkov, (0, dt), x0, args=(a, b)).y
            # # solution = solve_ivp(selkov, (0, dt), x.cpu(), args=(a, b))
            # solution = np.vectorize(solver)(x.cpu())

            outputs = np.zeros((x.shape[0], 2))
            for i in range(x.shape[0]):
                # solution = solve_ivp(selkov, (0, dt), x[i].cpu(), args=(a, b))
                solution = solve_ivp(selkov, (0, dt), x[i], args=(a, b))
                outputs[i] = solution.y[:, -1]

            return outputs

        return target_function

    def sample_tasks(self, num_tasks, return_specs=False):

        target_functions = []
        for i in range(num_tasks):
            a, b = self.environments[i]
            target_functions.append(self.get_target_function(a, b))

        if return_specs:
            return target_functions, a, b
        else:
            return target_functions

    def sample_datapoints(self, batch_size):
        """
        Sample random input/output pairs (e.g. for training an orcale)
        :param batch_size:
        :return:
        """

        amplitudes = torch.Tensor(np.random.uniform(self.amplitude_range[0], self.amplitude_range[1], batch_size))
        phases = torch.Tensor(np.random.uniform(self.phase_range[0], self.phase_range[1], batch_size))

        inputs = torch.rand((batch_size, self.num_inputs))
        inputs = inputs * (self.input_range[1] - self.input_range[0]) + self.input_range[0]
        inputs = inputs.view(-1)

        outputs = torch.sin(inputs - phases) * amplitudes
        outputs = outputs.unsqueeze(1)

        return torch.stack((inputs, amplitudes, phases)).t(), outputs
