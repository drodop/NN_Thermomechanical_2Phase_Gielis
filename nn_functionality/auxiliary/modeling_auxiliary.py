import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import sys
sys.path.append('../nn_lib/data_set_module')
sys.path.append('../nn_lib/nn_model')

import file_module as fm
import data_set as data_set_module
import nn_layer
import neural_network

# .......................................................... #
# Auxiliary Model Building function
# .......................................................... #
def build_model(ndim, ncm, nneuron, nhidden, activation):

    layer_vec = list()

    for ilayer in range(0,nhidden):
        # First hidden layer that takes as input the input vector
        if (ilayer == 0):
            li = nn_layer.nn_layer(nneuron, activation, (ndim,), None, None, None, None, None, None)
            li.construct_layer()
            layer_vec.append(li)
        else:
            # Rest hidden layers
            li = nn_layer.nn_layer(nneuron, activation, None, None, None, None, None, None, None)
            li.construct_layer()
            layer_vec.append(li)

    lout = nn_layer.nn_layer(ncm,    None,   None, None, None, None, None, None, None)
    lout.construct_layer()
    layer_vec.append(lout)

    m = neural_network.neural_network()    
    m.create_model_sequential(layer_vec)  

    return m   

# .......................................................... #
# Auxiliary Model Training function
# .......................................................... #
def train_model(model, optimizer, nepochs, xi_train, yi_train, xi_val, yi_val):

    # Manual training process
    train_loss_vec = list()
    val_loss_vec   = list()

    # Loop over the epochs
    for iepoch in range(0,nepochs):
        with tf.GradientTape() as tape:

            # Compute the model output
            yi_pred = model.model_(xi_train)

            # Loss for the regressor
            train_loss = tf.reduce_mean(tf.square(yi_train[:,0] - yi_pred[:,0])) + tf.reduce_mean(tf.square(yi_train[:,1] - yi_pred[:,1]))

        # Compute gradients
        gradients = tape.gradient(train_loss, model.model_.trainable_variables)   

        optimizer.apply_gradients(zip(gradients, model.model_.trainable_variables))

        if iepoch == 0:
            train_loss0 = train_loss

        train_loss_normalized = train_loss.numpy()/train_loss0

        train_loss_vec.append(train_loss_normalized)

        if train_loss_normalized < 2.0e-9:
            break

        if (xi_val.shape[0] > 0):
            # Compute the model output
            yi_pred_val = model.model_(xi_val)        
            # Validation loss
            val_loss = tf.reduce_mean(tf.square(yi_val[:,0] - yi_pred_val[:,0])) + tf.reduce_mean(tf.square(yi_val[:,1] - yi_pred_val[:,1]))            

            if iepoch == 0:
                val_loss0 = val_loss

            val_loss_normalized = val_loss.numpy()/val_loss0  

            val_loss_vec.append(val_loss_normalized)      

        # Print the loss every 100 epochs
        if iepoch % 10 == 0:
            print(f"Epoch {iepoch}/{nepochs}, Total Loss: {train_loss_normalized}")

    epochs = range(1, len(train_loss_vec) + 1)

    fig = plt.figure()
    subfig = fig.add_subplot(1,1,1)
    subfig.plot(epochs, train_loss_vec, 'r', label='Training loss')
    if (xi_val.shape[0] > 0):
        subfig.plot(epochs, val_loss_vec, 'b', label='Validation loss')
    subfig.legend()
    subfig.grid(True)

    return model   