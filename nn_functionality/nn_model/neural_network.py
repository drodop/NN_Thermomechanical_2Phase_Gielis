# .......................................................... #
# Functionality for Neural Networks
# Based on tensorflow-Keras
# D. Rodopoulos
# .......................................................... #
#
# .......................................................... #
# Basic modules
# .......................................................... #
#
import math
import pandas as pd
import matplotlib.pyplot as plt
import shap
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import tensorflow as tf
#
# .......................................................... #
# Imports of custom modules
# .......................................................... #
#
import nn_layer
#   
# .......................................................... #
# class Neural Network layer
# .......................................................... #
#
class functional_nn_layer:
    def __init__(self, layer_info):
        self.layer_ = layer_info

    layer_ = None

#   
# .......................................................... #
# class Neural Network - Sequential MLP
# .......................................................... #
#
class neural_network:
    def __init__(self):

        self.model_ = None

    model_ = None
    nlayer_ = None
    layer_vec_ = list()
    input_shape_ = 0
    output_shape_ = 0
    optimizer_ = 'adam'
    loss_ = 'mean_squared_error'
    metrics_ = ['acc']
    nepochs_ = 100
    validation_data_ = None
    input_data_ = None
    output_data_ = None
    # List that contains tf.Variable tensors as elements.
    # [Wij(0),bi(0),Wij(1),bi(1),....,Wij(nlayer),bi(nlayer)]
    trainable_variables_ = list()

    def create_model(self,ndim,ncm,layer_vec):
        
        # Creates a Keras model
        self.nlayer_ = len(layer_vec)

        self.layer_vec_ = layer_vec

        # Firstly define the input
        inp = tf.keras.Input(shape=(ndim,))

        lim1 = inp
        for ilayer in range(0,self.nlayer_):
            li = layer_vec[ilayer].layer_(lim1)
            lim1 = li

        self.model_ = tf.keras.Model(inp, li)        

    def create_model_sequential(self,layer_vec):
        
        # Creates a sequential Keras model
        self.nlayer_ = len(layer_vec)

        self.model_ = tf.keras.Sequential()

        for ilayer in range(0,self.nlayer_):
            self.model_.add(layer_vec[ilayer].layer_)

        self.layer_vec_ = layer_vec

    def create_vector_model(self,ndim,ncm,comp_layer_vec):
        
        # Creates a Keras vector model
        # comp_layer_vec is a list (size ncm) with elements the different lists of layers

        # Firstly define the input
        inp = tf.keras.Input(shape=(ndim,))        

        output = []

        # For each component assembly of the corresponding network component
        for icm in range(0,ncm):

            self.nlayer_ = len(comp_layer_vec[icm])

            self.layer_vec_ = comp_layer_vec[icm]
        
            lim1 = inp
            for ilayer in range(0,self.nlayer_):
                li = comp_layer_vec[icm][ilayer].layer_(lim1)
                lim1 = li
            
            output.append(li)
        
        self.model_ = tf.keras.Model(inp, tf.keras.layers.Concatenate(axis=1)(output))

    def create_manual_model(self,layer_vec):

        # Creates a manual model
        self.nlayer_ = len(layer_vec)

        self.layer_vec_ = layer_vec

    def print_model_summary(self):
        print(self.model_.summary())

    def get_layer(self,idx):

        return self.model_.layers[idx]
    
    def get_layer_weights(self,idx):

        # Returns Numpy matrices
        return self.model_.layers[idx].get_weights()[0]
    
    def get_layer_bias(self,idx):

        # Returns Numpy matrices
        return self.model_.layers[idx].get_weights()[1]    

    def compile_model(self, optimizer, loss , metrics):

        # Compile the model with a prescribed loss function
        self.model_.compile(optimizer, loss , metrics)
            
    def perform_fitting(self, xi, yi, epochs, xi_test, yi_test):

        # Perform fitting using Keras functionality

        self.input_data_ = xi
        self.output_data_ = xi
        self.nepochs_ = epochs
        self.validation_data_ = (xi_test, yi_test)

        md = self.model_.fit(xi, yi, epochs=self.nepochs_, validation_data=self.validation_data_)

        return md
    
    def perform_prediction(self, xi):

        # Perform Prediction using Keras functionality

        F = self.model_.predict(xi)

        return F
    
    def assing_layer_parameter_info(self):

        # Assign the NN parameters with respect to an already defined Keras model

        for ilayer in range(0, self.nlayer_):
            i1 = 0 + 2*ilayer
            i2 = 1 + 2*ilayer            
            self.layer_vec_[ilayer].Wij_ = self.model_.trainable_variables[i1]
            self.layer_vec_[ilayer].bi_  = self.model_.trainable_variables[i2]

            self.trainable_variables_.append(self.model_.trainable_variables[i1])
            self.trainable_variables_.append(self.model_.trainable_variables[i2])
    
    def perform_manual_prediction(self, xi):

        # Perform the prediction Manually
        # xi is a numpy array
        # Firstly, we convert in tf Tensor
        # The output is a tf Tensor

        xi_tf = tf.constant(xi, dtype=tf.float32)

        F_tf = xi_tf

        for ilayer in range(0,self.nlayer_):
            i1 = 0 + 2*ilayer
            i2 = 1 + 2*ilayer
            F_tf = self.layer_vec_[ilayer].factiv(tf.matmul(F_tf, self.model_.trainable_variables[i1]) + self.model_.trainable_variables[i2])          

        # for ilayer in range(0, self.nlayer_):

        #     Wij = self.get_layer_weights(ilayer)
        #     bi  = self.get_layer_bias(ilayer)

        #     Wij_tf = tf.constant(Wij, dtype=tf.float32)
        #     bi_tf  = tf.constant(bi, dtype=tf.float32)

        #     F_tf = self.layer_vec_[ilayer].factiv(tf.matmul(F_tf, Wij_tf) + bi_tf)

        F = F_tf

        return F

    def perform_manual_prediction2(self, xi):

        # Perform the prediction Manually
        # xi is a numpy array
        # Firstly, we convert in tf Tensor
        xi_tf = tf.constant(xi, dtype=tf.float32)

        F_tf = xi_tf

        for ilayer in range(0,self.nlayer_):

            Wij = self.layer_vec_[ilayer].Wij_
            bi  = self.layer_vec_[ilayer].bi_

            F_tf = self.layer_vec_[ilayer].factiv(tf.matmul(F_tf,Wij) + bi)          

        F = F_tf

        return F      
    
    def save_model(self, path):

        self.model_.save(path)

#   
# .......................................................... #
# Load model
# .......................................................... #
#
def load_model(path):

    m = neural_network()

    # Should copy the model in a better way. Make a copy function like copy_model()
    m.model_ = tf.keras.models.load_model(path)

    return m    