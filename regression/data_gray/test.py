## Check that there are no nan values in the data
import numpy as np

# define the path
train_data = np.load('train_data.npz')['X']
test_data = np.load('test_data.npz')['X']
adata_train = np.load('adapt_data.npz')['X']
adata_test = np.load('adapt_data_test.npz')['X']

# check for nan values
print('Train data: ', np.isnan(train_data).any())
print('Test data: ', np.isnan(test_data).any())
print('Adapt data: ', np.isnan(adata_train).any())
print('Adapt data test: ', np.isnan(adata_test).any())

# Check if data is finite
print('Train data: ', np.isfinite(train_data).all())
print('Test data: ', np.isfinite(test_data).all())
print('Adapt data: ', np.isfinite(adata_train).all())
print('Adapt data test: ', np.isfinite(adata_test).all())


## Print the max in absolute value of each dataset
print('Train data: ', np.max(np.abs(train_data)))
print('Test data: ', np.max(np.abs(test_data)))
print('Adapt data: ', np.max(np.abs(adata_train)))
print('Adapt data test: ', np.max(np.abs(adata_test)))


# Plot and save one trajectory from train_data
# import matplotlib.pyplot as plt
# plt.plot(train_data[0, :, 0], train_data[0, :, 1])