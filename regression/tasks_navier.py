import numpy as np
import torch
from scipy.integrate import solve_ivp
import math

class RegressionTasksNavier:
    """
    Same regression task as in Finn et al. 2017 (MAML)
    """

    def __init__(self, mode="train"):
        res = 32
        self.num_inputs = res*res*1
        self.num_outputs = res*res*1

        # self.a_param = [0.1]
        # self.b_param = [b for b in np.linspace(0.1, 1.0, 10)]

        # self.input_range = [-5, 5]
        if mode == "train" or mode == "valid":
            tt = torch.linspace(0, 1, res + 1)[0:-1]
            X, Y = torch.meshgrid(tt, tt)
            self.environments = [
                        {"f": 0.1 * (torch.sin(2 * math.pi * (X + Y)) + torch.cos(2 * math.pi * (X + Y))), "visc": 8e-4},
                        {"f": 0.1 * (torch.sin(2 * math.pi * (X + Y)) + torch.cos(2 * math.pi * (X + Y))), "visc": 9e-4},
                        {"f": 0.1 * (torch.sin(2 * math.pi * (X + Y)) + torch.cos(2 * math.pi * (X + Y))), "visc": 1.0e-3},
                        {"f": 0.1 * (torch.sin(2 * math.pi * (X + Y)) + torch.cos(2 * math.pi * (X + Y))), "visc": 1.1e-3},
                        {"f": 0.1 * (torch.sin(2 * math.pi * (X + Y)) + torch.cos(2 * math.pi * (X + Y))), "visc": 1.2e-3},
                    ]   
        elif mode == "adapt" or mode == "adapt_test":
            tt = torch.linspace(0, 1, res + 1)[0:-1]
            X, Y = torch.meshgrid(tt, tt)
            f = 0.1 * (torch.sin(2 * math.pi * (X + Y)) + torch.cos(2 * math.pi * (X + Y)))
            viscs = [8.5e-4, 9.5e-4, 1.05e-3, 1.15e-3]
            self.environments = [{"f": f, "visc": visc} for visc in viscs]

        self.mode = mode

    def get_input_range(self, size=100):
        return torch.linspace(self.input_range[0], self.input_range[1], steps=size).unsqueeze(1)

    def sample_inputs(self, batch_size, env_id, *args, **kwargs):

        if not hasattr(self, 'data'):
            if self.mode == "train":
                self.data = np.load('regression/data_navier/train_data.npz')
            elif self.mode == "valid":
                self.data = np.load('regression/data_navier/test_data.npz')
            elif self.mode == "adapt":
                self.data = np.load('regression/data_navier/adapt_data.npz')
            elif self.mode == "adapt_test":
                self.data = np.load('regression/data_navier/adapt_data_test.npz')

        X = self.data['X']
        # inputs = X[env_id, :, :-1, :].reshape((-1, self.num_inputs))
        inputs = X[env_id, :, 0, :].reshape((-1, self.num_inputs))
        return torch.Tensor(inputs), torch.Tensor(self.data['t'])

    def sample_targets(self, batch_size, env_id, *args, **kwargs):

        if not hasattr(self, 'data'):
            if self.mode == "train":
                self.data = np.load('regression/data_navier/train_data.npz')
            elif self.mode == "valid":
                self.data = np.load('regression/data_navier/test_data.npz')
            elif self.mode == "adapt":
                self.data = np.load('regression/data_navier/adapt_data.npz')
            elif self.mode == "adapt_test":
                self.data = np.load('regression/data_navier/adapt_data_test.npz')

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
