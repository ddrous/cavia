import torch
import torch.nn.functional as F
from torch import nn
from torchdiffeq import odeint
from torchdiffeq import odeint_adjoint
import numpy as np




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





class GroupSpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, groups=1):
        super().__init__()
        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2
        self.scale = 1 / (in_channels * out_channels)
        self.groups = groups
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(groups * in_channels, out_channels, self.modes1, self.modes2, 2))
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(groups * in_channels, out_channels, self.modes1, self.modes2, 2))

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, env, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, env, out_channel, x,y)
        return torch.einsum("beixy,eioxy->beoxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x)
        x_ft = x_ft.reshape(batchsize, self.groups, self.in_channels, x.size(-2), x.size(-1) // 2 + 1)
        # Multiply relevant Fourier modes
        weights1 = self.weights1.reshape(self.groups, self.in_channels, self.out_channels, self.modes1, self.modes2, 2)
        weights2 = self.weights2.reshape(self.groups, self.in_channels, self.out_channels, self.modes1, self.modes2, 2)
        out_ft = torch.zeros(batchsize, self.groups, self.out_channels, x.size(-2), x.size(-1) // 2 + 1, dtype=torch.cfloat,
                             device=x.device)
        out_ft[:, :, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :, :self.modes1, :self.modes2], torch.view_as_complex(weights1))
        out_ft[:, :, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :, -self.modes1:, :self.modes2], torch.view_as_complex(weights2))
        # Return to physical space
        out_ft = out_ft.reshape(batchsize, self.groups * self.out_channels, x.size(-2), x.size(-1) // 2 + 1)
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x

def batch_transform(batch, minibatch_size):
    # batch: b x c x t
    t = batch.shape[2:]
    new_batch = []
    for i in range(minibatch_size):
        sample = batch[i::minibatch_size]  # n_env x c x t
        sample = sample.reshape(-1, *t)
        new_batch.append(sample)
    return torch.stack(new_batch)  # minibatch_size x n_env * c x t

def batch_transform_inverse(new_batch, n_env):
    # new_batch: minibatch_size x n_env * c x t
    c = new_batch.size(1) // n_env
    t = new_batch.shape[2:]
    new_batch = new_batch.reshape(-1, n_env, c, *t)
    batch = []
    for i in range(n_env):
        sample = new_batch[:, i]  # minibatch_size x c x t
        batch.append(sample)
    return torch.cat(batch)  # b x c x t


class GroupFNO2d(nn.Module):
    def __init__(self, state_c, modes1=8, modes2=8, width=10, groups=1, nl='swish'):
        super().__init__()
        self.width = width
        self.groups = groups
        self.fc0 = nn.Conv2d((state_c + 3) * self.groups, self.width * self.groups, 1, groups=groups)
        self.conv0 = GroupSpectralConv2d(self.width, self.width, modes1, modes2, groups)
        self.conv1 = GroupSpectralConv2d(self.width, self.width, modes1, modes2, groups)
        self.conv2 = GroupSpectralConv2d(self.width, self.width, modes1, modes2, groups)
        self.conv3 = GroupSpectralConv2d(self.width, self.width, modes1, modes2, groups)
        self.w0 = nn.Conv2d(self.width * self.groups, self.width * self.groups, 1, groups=groups)
        self.w1 = nn.Conv2d(self.width * self.groups, self.width * self.groups, 1, groups=groups)
        self.w2 = nn.Conv2d(self.width * self.groups, self.width * self.groups, 1, groups=groups)
        self.w3 = nn.Conv2d(self.width * self.groups, self.width * self.groups, 1, groups=groups)
        self.a0 = GroupActivation(nl, groups=groups)
        self.a1 = GroupActivation(nl, groups=groups)
        self.a2 = GroupActivation(nl, groups=groups)
        self.a3 = GroupActivation(nl, groups=groups)
        self.fc1 = nn.Conv2d(self.width * self.groups, 16 * self.groups, 1, groups=groups)
        self.fc2 = nn.Conv2d(16 * self.groups, state_c * self.groups, 1, groups=groups)

        ## A new linear layer for the context parameters
        self.ctx_layer = nn.Linear(202, 1024)

    def forward(self, t, x, context):
        ## Reshape x into the correct shape
        x = x.view(-1, 1, 32, 32)

        ## Broadcast context to match the batch size
        context = self.ctx_layer(context)
        context = torch.broadcast_to(context, (x.shape[0], context.shape[0])).reshape(x.shape[0], 1, 32, 32)
        # context = context.view(-1, 1, 32, 32)

        # x:  batchsize x n_env * c x h x w
        # print("Original shapes: ", x.shape)

        minibatch_size = x.shape[0]
        x = batch_transform_inverse(x, self.groups)
        batchsize = x.shape[0]
        size_x, size_y = x.shape[-2], x.shape[-1]
        grid = self.get_grid(batchsize, size_x, size_y, x.device)


        # x = torch.cat((x, grid), dim=1)
        x = torch.cat((x, context, grid), dim=1)
        x = batch_transform(x, minibatch_size)

        # Lift with P
        x = self.fc0(x)
        # Fourier layer 0
        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = self.a0(x1 + x2)
        # Fourier layer 1
        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = self.a1(x1 + x2)
        # Fourier layer 2
        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = self.a2(x1 + x2)
        # Fourier layer 3
        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2

        # Projection with Q
        x = self.a3(self.fc1(x))
        x = self.fc2(x)

        # Reshape x into the correct shape
        x = x.view(-1, 1024)
        return x

    def get_grid(self, batchsize, size_x, size_y, device):
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, 1, size_x, 1).repeat([batchsize, 1, 1, size_y])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, 1, size_y).repeat([batchsize, 1, size_x, 1])
        return torch.cat((gridx, gridy), dim=1).to(device)





class CaviaModelFNO(nn.Module):
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
        super(CaviaModelFNO, self).__init__()

        self.device = device
        self.odefunc = GroupFNO2d(state_c=1, modes1=8, modes2=8, width=10, groups=1, nl='swish')

        # context parameters (note that these are *not* registered parameters of the model!)
        self.num_context_params = num_context_params
        self.context_params = None
        self.reset_context_params()

 
    def reset_context_params(self):
        self.context_params = torch.zeros(self.num_context_params).to(self.device)
        self.context_params.requires_grad = True

    def forward(self, x, t_eval):

        newodefunc = ODEFuncWithContext(self.odefunc, self.context_params)

        options = {}
        # options = {"first_step":50, "dtype":torch.float64}
        pred_y = odeint(newodefunc, x, t_eval, method='euler', rtol=1e-3, atol=1e-6, options=options)[:,...]

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


