import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import sys
sys.path.append('../nn_model')

import file_module as fm
import nn_layer
import neural_network

def prepare_2d_slice_data(x, y, ncm, variable_indeces, constant_indeces, dim_vec):

    N1 = dim_vec[variable_indeces[0]]
    N2 = dim_vec[variable_indeces[1]]
    N3 = dim_vec[constant_indeces[0]]
    # N4 = dim_vec[constant_indeces[1]]

    npt_slice = N1*N2
    xi_data = np.zeros((npt_slice,2))
    yi_data = np.zeros((npt_slice,ncm))

    idx = 0
    for i1 in range(0,N1):
        for i2 in range(0,N2):
            i3 = N3-1
            # i4 = N4-1
            
            ivec = 4*[0]
            Nvec = 4*[0]

            ivec[variable_indeces[0]] = i1
            ivec[variable_indeces[1]] = i2
            ivec[constant_indeces[0]] = N3-1
            # ivec[constant_indeces[1]] = N4-1

            Nvec[variable_indeces[0]] = N1
            Nvec[variable_indeces[1]] = N2
            Nvec[constant_indeces[0]] = N3
            # Nvec[constant_indeces[1]] = N4            

            # ipt = ivec[3] + Nvec[3]*( ivec[2] + Nvec[2]*(ivec[1] + Nvec[1]*ivec[0]) )
            ipt = ( ivec[2] + Nvec[2]*(ivec[1] + Nvec[1]*ivec[0]) )
            # ipt = (ivec[1] + Nvec[1]*ivec[0])
            
            # ipt = i4 + N4*( i3 + N3*(i2 + N2*i1) )
            xi_data[idx][0] = x[ipt][variable_indeces[0]]
            xi_data[idx][1] = x[ipt][variable_indeces[1]]

            for icm in range(0,ncm):
                yi_data[idx][icm] = y[ipt][icm]

            idx = idx + 1

    return xi_data, yi_data


def plot_results_thermal(xdata, ydata, fdata, x, y, f, xlabel, ylabel):

    # Plot the Young modulus
    fig = plt.figure()
    subfig = fig.add_subplot(1,1,1, projection='3d')

    subfig.scatter(xdata,ydata,fdata, label='data', color='blue')
    surf = subfig.plot_surface(x, y, f, cmap='viridis', alpha=0.5, label='Model prediction', edgecolor=None)
    fig.colorbar(surf) # Add a colorbar to a plot

    subfig.set_title('Thermal conductivity')
    subfig.set_xlabel(xlabel)
    subfig.set_ylabel(ylabel)

    plt.show()

def plot_results_elastic(xdata, ydata, fdata, x, y, f, xlabel, ylabel):

    # CAUTION: fdata and f are lists and each entry of the list is a matrix to be plotted

    # Plot the Young modulus
    fig = plt.figure()
    subfig = fig.add_subplot(1,1,1, projection='3d')

    subfig.scatter(xdata,ydata,fdata[0], label='data', color='blue')
    surf = subfig.plot_surface(x, y, f[0], cmap='viridis', alpha=0.5, label='Model prediction', edgecolor=None)
    fig.colorbar(surf) # Add a colorbar to a plot

    subfig.set_title('Young')
    subfig.set_xlabel(xlabel)
    subfig.set_ylabel(ylabel)

    plt.show()

    # Plot the Shear modulus
    fig2 = plt.figure()  
    subfig2 = fig2.add_subplot(1,1,1, projection='3d')

    subfig2.scatter(xdata,ydata,fdata[1], label='data', color='blue')
    surf = subfig2.plot_surface(x, y, f[1], cmap='viridis', alpha=0.5, label='Model prediction', edgecolor=None)
    fig2.colorbar(surf) # Add a colorbar to a plot

    subfig2.set_title('Shear')
    subfig2.set_xlabel(xlabel)
    subfig2.set_ylabel(ylabel)

    plt.show()