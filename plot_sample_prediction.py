#%%

import numpy as np
import numpy as jnp
import matplotlib.pyplot as plt

data = np.load("sample_predictions_cavia.npz")
X = data["X"][:, 0, :]
X_hat = data["X_hat"][:, 0, :]

X.shape

#%%
# data['X_hat'].shape
nb_mats = 1
nb_plot_timesteps = 10
cmap = "coolwarm"
test_length=10
res = 32

def vec_to_mats(vec_uv, res=32, nb_mats=2):
    """ Reshapes a vector into a set of 2D matrices """
    UV = jnp.split(vec_uv, nb_mats)
    return [jnp.reshape(UV[i], (res, res)) for i in range(nb_mats)]

fig, ax = plt.subplots(nrows=nb_mats*2, ncols=nb_plot_timesteps, figsize=(2*nb_plot_timesteps, 2*nb_mats*2))
for j in range(0, test_length, test_length//nb_plot_timesteps):
    gt_j = vec_to_mats(X[j], res, nb_mats)
    ncf_j = vec_to_mats(X_hat[j], res, nb_mats)
    for i in range(nb_mats):
        ax[2*i, j].imshow(gt_j[i], cmap=cmap, interpolation='bilinear', origin='lower')
        ax[2*i+1, j].imshow(ncf_j[i], cmap=cmap, interpolation='bilinear', origin='lower')


#%%
start = np.abs((X[0]-X_hat[0]).flatten())
print(start)
np.argsort(start)
start[206]