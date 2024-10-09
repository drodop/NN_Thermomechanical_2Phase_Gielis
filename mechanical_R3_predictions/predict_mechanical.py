import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import sys
sys.path.append('../nn_functionality/data_set_module')
sys.path.append('../nn_functionality/nn_model')
sys.path.append('../nn_functionality/auxiliary')

import file_module as fm
import data_set as data_set_module
import nn_layer
import neural_network

sys.path.append('../plots')
from plot_results import prepare_2d_slice_data
from plot_results import plot_results_elastic

# .......................................................... #
# Main program
# .......................................................... #

ndim = 3 # Number of dimensions of the input space
ncm = 2  # Number of components of the output (Young, Shear)

## Firstly, the training data are imported (for comparison reason)
# Read the data files
YoungData = fm.readDataFile("isothermal_young.txt", 'IsothermalYoung')
ShearData = fm.readDataFile("isothermal_shear.txt", 'IsothermalShear')

# Convert to numpy array
data_mat_young = np.array(YoungData)
data_mat_shear = np.array(ShearData)

N1 = 10  # Volume fraction points
N2 = 10  # shape points
N3 = 10  # material ratio points
N4 = 1   # base material points

npt = N1*N2*N3*N4

xi = np.zeros((npt,ndim))
yi = np.zeros((npt,ncm))

ii = 0

nparam = 8

E_refval = 69e9 # reference base Young (Aluminium)

for i1 in range(0,N1):
    for i2 in range(0,N2):
        for i3 in range(0,N3):
            i4 = 0

            ipt = i4 + N4*( i3 + N3*(i2 + N2*i1) )
            xi[ipt][0] = data_mat_young[ipt][0] # Vf
            xi[ipt][1] = data_mat_young[ipt][5] # n
            xi[ipt][2] = data_mat_young[ipt][1] # Young ratio

            yi[ipt][0] = data_mat_young[ipt][nparam] / E_refval
            yi[ipt][1] = data_mat_shear[ipt][nparam] / E_refval


## Load the model
m1 = neural_network.load_model('./')

## Plot the slices
# Define the grid of the test data
# The following correspondance is made:
#   Vf -> x1
#   n  -> x2
#   rE -> x3

maxVf= np.max(xi[:,0])
maxn = np.max(xi[:,1])
maxr = np.max(xi[:,2])

minVf= np.min(xi[:,0])
minn = np.min(xi[:,1])
minr = np.min(xi[:,2])

x1_eval = np.linspace(minVf,maxVf,20)
x2_eval = np.linspace(minn,maxn,20)
x3_eval = np.linspace(minr,maxr,20)

# Make the (x3 = c) slice - Fixed Material ratio
X1, X2 = np.meshgrid(x1_eval, x2_eval)
X1_unorm, X2_unorm = np.meshgrid(x1_eval, x2_eval)

X1_flat = X1.flatten()
X2_flat = X2.flatten()

Xi_eval = np.column_stack((X1_flat,X2_flat, maxr*np.ones(X2_flat.shape)))

## Evaluate the Neural network
F = m1.perform_prediction(Xi_eval)
F1 = F[:,0].reshape(X1.shape)
F2 = F[:,1].reshape(X1.shape)

xi_data, yi_data = prepare_2d_slice_data(xi, yi, ncm, [0,1], [2,], (N1,N2,N3,N4))

plot_results_elastic(xi_data[:,0], xi_data[:,1], (yi_data[:,0],yi_data[:,1]), X1_unorm, X2_unorm, (F1,F2), 'Vf', 'n')

# Make the (x1 = c) slice - Fixed volume fraction
X2, X3 = np.meshgrid(x2_eval, x3_eval)
X2_unorm, X3_unorm = np.meshgrid(x2_eval, x3_eval)

X2_flat = X2.flatten()
X3_flat = X3.flatten()

Xi_eval = np.column_stack((maxVf*np.ones(X2_flat.shape), X2_flat, X3_flat))

## Evaluate the Neural network
F = m1.perform_prediction(Xi_eval)
F1 = F[:,0].reshape(X1.shape)
F2 = F[:,1].reshape(X1.shape)

xi_data, yi_data = prepare_2d_slice_data(xi, yi, ncm, [1,2], [0,], (N1,N2,N3,N4))

plot_results_elastic(xi_data[:,0], xi_data[:,1], (yi_data[:,0],yi_data[:,1]), X2_unorm, X3_unorm, (F1,F2), 'n', 'rE')

# Make the (x2 = c) slice - Fixed Geometry

X1, X3 = np.meshgrid(x1_eval, x3_eval)
X1_unorm, X3_unorm = np.meshgrid(x1_eval, x3_eval)

X1_flat = X1.flatten()
X3_flat = X3.flatten()

Xi_eval = np.column_stack((X1_flat, maxn*np.ones(X2_flat.shape), X3_flat))

## Evaluate the Neural network
F = m1.perform_prediction(Xi_eval)
F1 = F[:,0].reshape(X1.shape)
F2 = F[:,1].reshape(X1.shape)

xi_data, yi_data = prepare_2d_slice_data(xi, yi, ncm, [0,2], [1,], (N1,N2,N3,N4))

plot_results_elastic(xi_data[:,0], xi_data[:,1], (yi_data[:,0],yi_data[:,1]), X1_unorm, X3_unorm, (F1,F2), 'Vf', 'rE')        
