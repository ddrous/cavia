import numpy as np
import torch
from scipy.integrate import solve_ivp

class RegressionTasksBrussel:
    """
    Same regression task as in Finn et al. 2017 (MAML)
    """

    def __init__(self, mode="train"):
        self.num_inputs = 8*8*2
        self.num_outputs = 8*8*2

        # self.a_param = [0.1]
        # self.b_param = [b for b in np.linspace(0.1, 1.0, 10)]

        # self.input_range = [-5, 5]
        if mode == "train" or mode == "valid":
            As = [0.75, 1., 1.25]
            Bs = [3.25, 3.5, 3.75]
            self.environments = [{"A": A, "B": B, "Du": 1.0, "Dv": 0.1} for A in As for B in Bs]
        elif mode == "adapt" or mode == "adapt_test":
            As = [0.875, 1.125, 1.375]
            Bs = [3.125, 3.375, 3.625, 3.875]
            self.environments = [{"A": A, "B": B, "Du": 1.0, "Dv": 0.1} for A in As for B in Bs]
        self.mode = mode

    def get_input_range(self, size=100):
        return torch.linspace(self.input_range[0], self.input_range[1], steps=size).unsqueeze(1)

    def sample_inputs(self, batch_size, env_id, *args, **kwargs):

        if not hasattr(self, 'data'):
            if self.mode == "train":
                self.data = np.load('regression/data_brussel/train_data.npz')
            elif self.mode == "valid":
                self.data = np.load('regression/data_brussel/test_data.npz')
            elif self.mode == "adapt":
                self.data = np.load('regression/data_brussel/adapt_data.npz')
            elif self.mode == "adapt_test":
                self.data = np.load('regression/data_brussel/adapt_data_test.npz')

        X = self.data['X']
        # inputs = X[env_id, :, :-1, :].reshape((-1, self.num_inputs))
        inputs = X[env_id, :, 0, :].reshape((-1, self.num_inputs))
        return torch.Tensor(inputs), torch.Tensor(self.data['t'])

    def sample_targets(self, batch_size, env_id, *args, **kwargs):

        if not hasattr(self, 'data'):
            if self.mode == "train":
                self.data = np.load('regression/data_brussel/train_data.npz')
            elif self.mode == "valid":
                self.data = np.load('regression/data_brussel/test_data.npz')
            elif self.mode == "adapt":
                self.data = np.load('regression/data_brussel/adapt_data.npz')
            elif self.mode == "adapt_test":
                self.data = np.load('regression/data_brussel/adapt_data_test.npz')

        X = self.data['X']
        outputs = X[env_id, :, :, :]
        # outputs = X[env_id, :, 1:, :].reshape((-1, self.num_inputs))
        return torch.Tensor(outputs).permute((1,0,2)), torch.Tensor(self.data['t'])

    def sample_task(self):
        pass

    @staticmethod
    def get_target_function(a, b):
        pass

    def sample_tasks(self, num_tasks, return_specs=False):
        pass

    def sample_datapoints(self, batch_size):
        """
        Sample random input/output pairs (e.g. for training an orcale)
        :param batch_size:
        :return:
        """
        pass
