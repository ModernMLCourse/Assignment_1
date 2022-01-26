"""
EE468: Neural Networks and Deep Learning class.
First assignment-- Regression
Instructor notes:
-----------------
This script is meant as a guid for students to learn Python and do
Assignment 1 of the course. It has missing lines that students are
expected to fill in. Hence, you should read every line carefully and
make use of the notes scattered throughout this script.

And, yeah, try to enjoy it... ;-)
--------------
Author: Muhammad Alrabeiah
Date: Jan. 2022
"""

# Imports
# -------
import os # A module for path, file, and directory manipulations
import numpy as np # Computations
import scipy.io as sio # Loading matlab data files
import matplotlib.pyplot as plt # Visualization


# Let's start by loading data and doing some visualization
# --------------------------------------------------------

data_file = './assig_1_dataset_1.mat' # Path pointing to your data file
D =  # TODO: You should load the .mat data. Check out the function ``loadmat'' from scipy.io. Data will be loaded into a Python dictionary. Read about dictionaries!
D_keys = D.keys() # Get names of variables in D (called dictionary keys)
print(D_keys) # Print the list of names. NOTE: ignore names enclosed in '__whatever__' 
              # NOTE: x has observed variable values (inputs) and y_n has desired response values

obs_vars = D['x'][:,:300] # Reading the first 300 inputs as training observed variables
dsr_rsps = # TODO: Read the corresponding desired training responses from y_n
obs_vars_val = D['x'][:,300:] # Reading the last 100 inputs as validation observed variables
dsr_rsps_val = # TODO: Read the corresponding desired validation responses from y_n

#### Milestone ####
# Not how ``:'' is used to read first and last chunck of data from a numpy array.
# Reading a chunck of data from a numpy array is called ``slicing'' in Python
###################

plt.figure(1,figsize=[10,8]) # Define a figure
plt.scatter(obs_vars, dsr_rsps, label='training data') # Plotting the training data with a legend and a grid
plt.xlabel('Observed variable')
plt.ylabel('Desired response')
plt.grid()
plt.legend()

# Fitting a straight line
# -----------------------
X = np.concatenate([np.ones(obs_vars.shape), obs_vars],axis=0) # Construct design matrix (Check: https://numpy.org/doc/stable/reference/generated/numpy.concatenate.html)
X = # TODO: Take the transpose of X. Check out the function np.transpose or ``.T''
y = # TODO: Take the transpose of y


inv_XX = np.linalg.inv(  ) # TODO: What does linalginv do? what should be its argument?
P = np.matmul(inv_XX, X.T)
w_m1 = # TODO: compute the weights of the model

y_hat1 = np.matmul( X, w_m1 ) # Compute the predictions of the model

plt.figure(2,figsize=(10,8))
plt.scatter(obs_vars,dsr_rsps,label='Data points')
plt.plot(obs_vars.T, y_hat1, '-r', label='Linear model') # Why did we take .T of obs_vars?
plt.xlabel('Observed variable')
plt.ylabel('Response')
plt.legend()
plt.grid()

# Fitting with basis functions
# ----------------------------
N = # TODO: this is the number of basis function you should use. Try numbers from 2 to 4 (See theoretical part of the)
X = np.ones((N+1,obs_vars.shape[1])) # Pre-allocate memory to the design matrix (VERY IMPORTANT AND GOOD PRACTICE)
for i in range(1,N+1):
    X[i,:] = # TODO: compute each row of basis functions. HINT: in Python ``x**i'' means every element in x is raised to the power of i. Make use of this!

X = X.T

#TODO: use the design matrix X to find optimal w_m2 for model 2
inv_XX = # TODO
P =  # TODO
w_m2 = # TODO

y_hat2 =  # TODO: Compute predictions

plt.figure(3,figsize=(10,8))
plt.scatter(obs_vars,dsr_rsps,label='Data points')
plt.plot(obs_vars.T, y_hat1, '-r', label='No-basis')
plt.plot(obs_vars.T, y_hat2, '-k', label='Polynomial basis')
plt.xlabel('Observed variable')
plt.ylabel('Response')
plt.legend()
plt.grid()

# Test the two models:
# --------------------
# TODO: Use your models (w_m1 and w_m2) to get predictions for obs_var_val.
#       Then, calculate the MSE loss for each one. THIS IS A BOUNS



'''
Question:
What happens when we change the vale of N from 2 to 4? What should be the best value for N?
Answer:

--------------------
Congrats. You have build two ML models!!
'''



