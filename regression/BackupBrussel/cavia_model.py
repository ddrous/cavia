import torch
import torch.nn.functional as F
from torch import nn
from torchdiffeq import odeint
from torchdiffeq import odeint_adjoint




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
            # x = F.relu(self.fc_layers[k](x))
            x = F.softplus(self.fc_layers[k](x))
        y = self.fc_layers[-1](x)

        return y






class BetaDeltaModel(nn.Module):
    def __init__(self, beta, delta):
        super(BetaDeltaModel, self).__init__()
        # beta, delta = torch.tensor(beta, requires_grad=True), torch.tensor(delta, requires_grad=True)
        # self.parameters = nn.ParameterList([beta, delta])

        # self.beta, self.delta = torch.tensor(beta, requires_grad=True), torch.tensor(delta, requires_grad=True)

        ## One linear layer that takes in the context parameters
        self.layer = nn.Linear(16, 2)


    def forward(self, ctx):
        # return self.parameters()
        # return self.beta, self.delta
        return torch.nn.Sigmoid()(self.layer(ctx))




class CaviaModel(nn.Module):
    """
    Feed-forward vector field with context parameters.
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

        ## Convolutional layers
        # self.odefunc = GroupConv(2, hidden_c=172, groups=1, factor=1.0, nl="swish", size=64, kernel_size=3)

        # self.betadel = BetaDeltaModel(0.5, 0.5)


        # context parameters (note that these are *not* registered parameters of the model!)
        self.num_context_params = num_context_params
        self.context_params = None
        self.reset_context_params()

        # self.parameters = nn.ParameterList([self.betadel.beta, self.betadel.delta, self.context_params])

 
    def reset_context_params(self):
        self.context_params = torch.zeros(self.num_context_params).to(self.device)
        self.context_params.requires_grad = True

    def forward(self, x, t_eval):

        # print(x.shape, t_eval.shape)

        def newodefunc(t,x):
            return self.odefunc(t,x,self.context_params,)
        pred_y = odeint(newodefunc, x, t_eval, method='dopri5')[:,...]
        # pred_y = odeint_adjoint(newodefunc, x, t_eval, method='dopri5')[:,...]

        # beta, delta = self.betadel(self.context_params)
        # def lotka_voltera(t, y):
        #     y = y[0,...]
        #     dx = 0.5*y[0] - beta * y[0] * y[1]
        #     dy = delta * y[0] * y[1] - 0.5*y[1]
        #     return torch.stack([dx, dy])
        # pred_y = odeint(lotka_voltera, x, t_eval, method='rk4')[:,...]

        return pred_y















from functools import partial
nls = {'relu': partial(nn.ReLU),
       'sigmoid': partial(nn.Sigmoid),
       'tanh': partial(nn.Tanh),
       'selu': partial(nn.SELU),
       'softplus': partial(nn.Softplus),
    #    'swish': partial(Swish),
    #    'sinus': partial(Sinus),
       'elu': partial(nn.ELU)}

class GroupSwish(nn.Module):
    def __init__(self, groups):
        super().__init__()
        self.beta = nn.Parameter(torch.tensor([0.5 for _ in range(groups)]))
        self.groups = groups

    def forward(self, x):
        n_ch_group = x.size(1) // self.groups
        t = x.shape[2:]
        x = x.reshape(-1, self.groups, n_ch_group, *t)
        beta = self.beta.view(1, self.groups, 1, *[1 for _ in t])
        return (x * torch.sigmoid_(x * F.softplus(beta))).div_(1.1).reshape(-1, self.groups * n_ch_group, *t)

class GroupActivation(nn.Module):
    def __init__(self, nl, groups=1):
        super().__init__()
        self.groups = groups
        if nl == 'swish':
            self.activation = GroupSwish(groups)
        else:
            self.activation = nls[nl]()

    def forward(self, x):
        return self.activation(x)

class GroupConv(nn.Module):
    def __init__(self, state_c, hidden_c=2, groups=1, factor=1.0, nl="swish", size=64, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        self.out_c = state_c
        self.factor = factor
        self.hidden_c = hidden_c
        self.size = size
        self.net = nn.Sequential(
            nn.Conv2d(state_c * groups+1, hidden_c * groups, kernel_size=kernel_size, padding=padding, padding_mode='circular', groups=groups),
            GroupActivation(nl, groups=groups),
            nn.Conv2d(hidden_c * groups, hidden_c * groups, kernel_size=kernel_size, padding=padding, padding_mode='circular', groups=groups),
            GroupActivation(nl, groups=groups),
            nn.Conv2d(hidden_c * groups, hidden_c * groups, kernel_size=kernel_size, padding=padding, padding_mode='circular', groups=groups),
            GroupActivation(nl, groups=groups),
            nn.Conv2d(hidden_c * groups, hidden_c * groups, kernel_size=kernel_size, padding=padding, padding_mode='circular', groups=groups),
            GroupActivation(nl, groups=groups),
            nn.Conv2d(hidden_c * groups, hidden_c * groups, kernel_size=kernel_size, padding=padding, padding_mode='circular', groups=groups),
            GroupActivation(nl, groups=groups),
            nn.Conv2d(hidden_c * groups, hidden_c * groups, kernel_size=kernel_size, padding=padding, padding_mode='circular', groups=groups),
            GroupActivation(nl, groups=groups),
            nn.Conv2d(hidden_c * groups, hidden_c * groups, kernel_size=kernel_size, padding=padding, padding_mode='circular', groups=groups),
            GroupActivation(nl, groups=groups),
            nn.Conv2d(hidden_c * groups, state_c * groups, kernel_size=kernel_size, padding=padding, padding_mode='circular', groups=groups)
        )
        self.flatten = nn.Flatten()

    def forward(self, t, x, context):
        
        ## Reshape the input to the correct shape: 2 channels of 32x32
        size = 8 ## 8 for brussel, 32 for gray-scott

        x = x.view(-1, 2, size, size)

        ## repeat the context to match the batch size
        # context = context.repeat(x.shape[0], -1)
        # context = torch.broadcast_to(context, (x.shape[0], context.shape[0])).view(x.shape[0], 1, 32, 32)
        context = torch.broadcast_to(context, (x.shape[0], context.shape[0])).reshape(x.shape[0], 1, size, size)
        # context = context.reshape(x.shape[0], 1, 32, 32)

        ## The context is stacked as one channel of the 32x32 image
        # context = context

        # if x.shape[0] >1:
        #     print(x.shape, context.shape)

        x = torch.cat([x, context], dim=1)

        x = self.net(x) 

        # print(x.flatten().shape)
        ## Return flattene output
        return self.flatten(x)



class ODEFuncWithContext(nn.Module):
    """
    Feed-forward neural network with context parameters.
    """

    def __init__(self, odefunc, context):
        super(ODEFuncWithContext, self).__init__()

        self.odefunc = odefunc
        self.context = context

    def forward(self, t, x):
        return self.odefunc(t, x, self.context)




class CaviaModelConv(nn.Module):
    """
    Convolutional vector field with context parameters.
    """

    def __init__(self,
                 n_in,
                 n_out,
                 num_context_params,
                 n_hidden,
                 device
                 ):
        super(CaviaModelConv, self).__init__()


        self.device = device
        # Convolutional layers
        self.odefunc = GroupConv(2, hidden_c=46, groups=1, factor=1e-3, nl="swish", size=64, kernel_size=3)

        # self.betadel = BetaDeltaModel(0.5, 0.5)


        # context parameters (note that these are *not* registered parameters of the model!)
        self.num_context_params = num_context_params
        self.context_params = None
        self.reset_context_params()
        # self.newodefunc = ODEFuncWithContext(self.odefunc, self.context_params)

        # self.parameters = nn.ParameterList([self.betadel.beta, self.betadel.delta, self.context_params])

 
    def reset_context_params(self):
        self.context_params = torch.zeros(self.num_context_params).to(self.device)
        self.context_params.requires_grad = True

    def forward(self, x, t_eval):

        # print(x.shape, t_eval.shape)

        # def newodefunc(t,x):
        #     return self.odefunc(t,x,self.context_params,)
        # pred_y = odeint(newodefunc, x, t_eval, method='dopri5')[:,...]
        # # pred_y = odeint_adjoint(newodefunc, x, t_eval, method='dopri5', adjoint_params=())[:,...]


        newodefunc = ODEFuncWithContext(self.odefunc, self.context_params)
        # self.newodefunc.context = self.context_params
        # self.newodefunc.forward = lambda t, x: self.odefunc(t, x, self.context_params)

        # options=dict(step_size=40)
        # pred_y = odeint(newodefunc, x, t_eval, method='euler', rtol=1e-3, atol=1e-6, options=options)[:,...]
        # pred_y = odeint_adjoint(newodefunc, x, t_eval, method='dopri5')[:,...]


        # t_eval = torch.Tensor([0, 0., 1.]).to(self.device)
        # t_eval = torch.linspace(0, 100, 10).to(self.device)

        options = {"first_step":10, "dtype":torch.float64, "step_size":40}
        # options = {"first_step":50, "dtype":torch.float64}
        pred_y = odeint(newodefunc, x, t_eval, method='dopri5', rtol=1e-3, atol=1e-6, options=options)[:,...]
        # # pred_y = odeint(newodefunc, x, t_eval, method='dopri5', options=options)[:,...]


        ## In a for loop, integrate fron t[i] to t[i+1] and concatenate the results
        # pred_y = [x]
        # for i in range(len(t_eval)-1):
        #     pred_y.append(odeint(newodefunc, x, t_eval[i:i+2], method='dopri5', rtol=1e-3, atol=1e-6, options=options)[-1,...])
        # pred_y = torch.stack(pred_y, dim=0)





        return pred_y * 1e-1








# import torch
# import torch.nn.functional as F
# from torch import nn


# class CaviaModel(nn.Module):
#     """
#     Feed-forward neural network with context parameters.
#     """

#     def __init__(self,
#                  n_in,
#                  n_out,
#                  num_context_params,
#                  n_hidden,
#                  device
#                  ):
#         super(CaviaModel, self).__init__()

#         self.device = device

#         # fully connected layers
#         self.fc_layers = nn.ModuleList()
#         self.fc_layers.append(nn.Linear(n_in + num_context_params, n_hidden[0]))
#         for k in range(len(n_hidden) - 1):
#             self.fc_layers.append(nn.Linear(n_hidden[k], n_hidden[k + 1]))
#         self.fc_layers.append(nn.Linear(n_hidden[-1], n_out))

#         # context parameters (note that these are *not* registered parameters of the model!)
#         self.num_context_params = num_context_params
#         self.context_params = None
#         self.reset_context_params()

#     def reset_context_params(self):
#         self.context_params = torch.zeros(self.num_context_params).to(self.device)
#         self.context_params.requires_grad = True

#     def forward(self, x):

#         # concatenate input with context parameters
#         x = torch.cat((x, self.context_params.expand(x.shape[0], -1)), dim=1)

#         for k in range(len(self.fc_layers) - 1):
#             x = F.relu(self.fc_layers[k](x))
#         y = self.fc_layers[-1](x)

#         return y


