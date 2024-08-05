#%%


# problem = 'Brussel'
## Open the nohup.log time and read its contents
with open('nohup.log', 'r') as file:
    nohup_log = file.read()


## Sample line from the nohup.file is: 
    # Iter 0    - time: 3     - [train] loss: 1.246  (+/-0.219 ) - [valid] loss: 1.1839 (+/-0.2074) - [test] loss: 1.5598 (+/-0.5549)


## Use regex to extract the following: itern, train_loss valied_loss, and test_loss
import re

# Define the regex pattern
# pattern = r'Iter (\d+)\s+- time: \d+\s+- \[train\] loss: (\d+\.\d+)\s+\(\+\/-(\d+\.\d+)\)\s+- \[valid\] loss: (\d+\.\d+)\s+\(\+\/-(\d+\.\d+)\)\s+- \[test\] loss: (\d+\.\d+)\s+\(\+\/-(\d+\.\d+)\)'

## New patter fater the confidences were removed
# pattern = r'Iter (\d+)\s+- time: \d+\s+- \[train\] loss: (\d+\.\d+)\s+- \[valid\] loss: (\d+\.\d+)\s+- \[test\] loss: (\d+\.\d+)'


## New pattern with a line that looks like Iter 94   - time: 5     - [train]: 1.9977 - [valid]: 2.3995 - [adapt]: 2.1516 - [adapt_test]: 2.4253 - [adapt_test_per_env]: [2.0153, 2.352, 2.8543, 3.5427, 2.0641, 2.1897, 2.404, 2.8404, 2.2243, 2.1452, 2.1611, 2.3103]
pattern = r'Iter (\d+)\s+- time: \d+\s+- \[train\]: (\d+\.\d+)\s+- \[valid\]: (\d+\.\d+)\s+- \[adapt\]: (\d+\.\d+)\s+- \[adapt_test\]: (\d+\.\d+)\s+- '


# Find all the matches
matches = re.findall(pattern, nohup_log)

# Display the first 5 matches
matches[:5]


## Place the matches in a pandas dataframe
import pandas as pd

# Create a dataframe from the matches
# df = pd.DataFrame(matches, columns=['iter', 'train_loss', 'train_conf', 'valid_loss', 'valid_conf', 'test_loss', 'test_conf'])

## New dataframe after the confidences were removed
df = pd.DataFrame(matches, columns=['iter', 'train_loss', 'test_loss', 'adapt_train', 'adapt_test'])

## Convert the columns to numeric
df = df.apply(pd.to_numeric, errors='ignore')


# Display the first 5 rows of the dataframe
df.head()
df.dtypes

# %%

## Plot the train, valid, and test loss
import matplotlib.pyplot as plt

# Set the figure size
plt.figure(figsize=(10, 6))
# plt.plot(df['iter'], df['train_loss'], label='train loss')
# plt.plot(df['iter'], df['valid_loss'], label='valid loss')
# plt.plot(df['iter'], df['test_loss'], label='test loss')

## New plot the four columns
plt.plot(df['iter'], df['train_loss'], "b-", label='train loss')
plt.plot(df['iter'], df['test_loss'], "--", color="skyblue", label='test loss')
plt.plot(df['iter'], df['adapt_train'], "r-", label='adapt train loss')
plt.plot(df['iter'], df['adapt_test'], "--", color="orange", label='adapt test loss')

plt.xlabel('Iteration')

plt.legend()

plt.yscale('log')
plt.ylabel('Loss')
plt.title('Train, Valid, and Test Loss')


## Save the plot
# plt.savefig('losses_Brussel.png')


# %%
print(df.tail())

## Dumpp the dataframe to a csv file
# df.to_csv('losses_Brussel.csv', index=False)