import torch
import torch.nn.functional as F
from torch import nn
from torchdiffeq import odeint




class CaviaModelOld(nn.Module):
    """
    Feed-forward neural network with context parameters.
    """

    def __init__(self,
                 n_in,
                 n_out,
                 num_context_params,
                 n_hidden,
                 device
                 ):
        super(CaviaModelOld, self).__init__()

        self.device = device

        # fully connected layers
        self.fc_layers = nn.ModuleList()
        self.fc_layers.append(nn.Linear(n_in + num_context_params, n_hidden[0]))
        for k in range(len(n_hidden) - 1):
            self.fc_layers.append(nn.Linear(n_hidden[k], n_hidden[k + 1]))
        self.fc_layers.append(nn.Linear(n_hidden[-1], n_out))

        # context parameters (note that these are *not* registered parameters of the model!)
        self.num_context_params = num_context_params
        self.context_params = None
        self.reset_context_params()

    def reset_context_params(self):
        self.context_params = torch.zeros(self.num_context_params).to(self.device)
        self.context_params.requires_grad = True

    def forward(self, x):

        # concatenate input with context parameters
        x = torch.cat((x, self.context_params.expand(x.shape[0], -1)), dim=1)

        for k in range(len(self.fc_layers) - 1):
            x = F.relu(self.fc_layers[k](x))
        y = self.fc_layers[-1](x)

        return y



class ODEFunc(nn.Module):
    """
    Feed-forward neural network with context parameters.
    """

    def __init__(self,
                 n_in,
                 n_out,
                 num_context_params,
                 n_hidden,
                 device
                 ):
        super(ODEFunc, self).__init__()

        self.device = device

        # fully connected layers
        self.fc_layers = nn.ModuleList()
        self.fc_layers.append(nn.Linear(n_in + num_context_params, n_hidden[0]))
        for k in range(len(n_hidden) - 1):
            self.fc_layers.append(nn.Linear(n_hidden[k], n_hidden[k + 1]))
        self.fc_layers.append(nn.Linear(n_hidden[-1], n_out))

    def forward(self, t, x, context_params):

        # concatenate input with context parameters
        x = torch.cat((x, context_params.expand(x.shape[0], -1)), dim=1)

        for k in range(len(self.fc_layers) - 1):
            x = F.relu(self.fc_layers[k](x))
        y = self.fc_layers[-1](x)

        return y


# class CaviaModel(nn.Module):
#     """
#     Feed-forward neural network with context parameters.
#     """

#     def __init__(self,
#                  n_in,
#                  n_out,
#                  num_context_params,
#                  n_hidden,
#                  dt,
#                  device
#                  ):
#         super(CaviaModel, self).__init__()

#         self.device = device

#         # fully connected layers
#         self.dt = dt
#         self.odefunc = ODEFunc(n_in, n_out, num_context_params, n_hidden, device)

#         # context parameters (note that these are *not* registered parameters of the model!)
#         self.num_context_params = num_context_params
#         self.context_params = None
#         self.reset_context_params()

#     def reset_context_params(self):
#         self.context_params = torch.zeros(self.num_context_params).to(self.device)
#         self.context_params.requires_grad = True

#     def forward(self, x):

#         # concatenate input with context parameters
#         # pred_y = odeint(self.odefunc, x, self.dt, self.context_params).to(self.device)

#         t = torch.tensor([0., self.dt]).to(self.device)
#         def newodefunc(t,x):
#             return self.odefunc(t,x,self.context_params,)
#         pred_y = odeint(newodefunc, x, t,  method='dopri5')[-1,...]

#         # pred_y = self.odefunc(0, x, self.context_params)

#         return pred_y



class CaviaModel(nn.Module):
    """
    Feed-forward neural network with context parameters.
    """

    def __init__(self,
                 n_in,
                 n_out,
                 num_context_params,
                 n_hidden,
                 device
                 ):
        super(CaviaModel, self).__init__()

        self.device = device

        # fully connected layers
        self.odefunc = ODEFunc(n_in, n_out, num_context_params, n_hidden, device)

        # context parameters (note that these are *not* registered parameters of the model!)
        self.num_context_params = num_context_params
        self.context_params = None
        self.reset_context_params()

    def reset_context_params(self):
        self.context_params = torch.zeros(self.num_context_params).to(self.device)
        self.context_params.requires_grad = True

    def forward(self, x, t_eval):

        # concatenate input with context parameters
        # pred_y = odeint(self.odefunc, x, self.dt, self.context_params).to(self.device)

        def newodefunc(t,x):
            return self.odefunc(t,x,self.context_params,)
        pred_y = odeint(newodefunc, x, t_eval,  method='dopri5')[:,...]

        # pred_y = self.odefunc(0, x, self.context_params)

        return pred_y
