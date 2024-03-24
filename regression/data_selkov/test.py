#%%

## Open the train npz data and print the 'X' key
import numpy as np
data = np.load('train_data.npz')
print(data['X'][0])

print(data['X'].shape)

## Small experiment with the seed in numpy
# import numpy as np
# np.random.seed(0)
# print(np.random.rand(5))

# np.random.seed(1)
# print(np.random.rand(5))

# ##-------

# np.random.seed(0)
# print(np.random.rand(5))

# print(np.random.rand(5))

# %%
