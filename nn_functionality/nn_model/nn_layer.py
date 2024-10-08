# .......................................................... #
# Functionality for information contained in a Neural Network layer
# D. Rodopoulos
# .......................................................... #
#
# .......................................................... #
# Basic modules
# .......................................................... #
#
import numpy as np
import tensorflow as tf
#
# .......................................................... #
# Class nn_layer: Information about a Neural Network layer
# .......................................................... #
#
class nn_layer:
    def __init__(self, nneuron, activation=None, input_shape=None, 
                 kernel_initializer='glorot_uniform', bias_initializer='zeros', 
                 kernel_regularizer=None, bias_regularizer=None,
                 kernel_constraint=None, bias_constraint=None):
        self.nneuron_ = nneuron
        self.activation_ = activation
        self.input_shape_ = input_shape
        self.kernel_initializer_ = kernel_initializer
        self.bias_initializer_ = bias_initializer
        self.kernel_regularizer_ = kernel_regularizer
        self.bias_regularizer_ = bias_regularizer        
        self.kernel_constraint_ = kernel_constraint
        self.bias_constraint_ = bias_constraint
        
    layer_ = None

    nneuron_ = 50
    activation_ = None
    input_shape_ = None
    kernel_initializer_ = 'glorot_uniform'
    bias_initializer_ = 'zeros'
    kernel_regulizer_ = None
    bias_regulizer_ = None    
    kernel_constraint_ = None
    bias_constraint_ = None
    Wij_ = None
    bi_  = None        
    
    def construct_layer(self):
    
        if self.input_shape_ is not None:
            self.layer_ = tf.keras.layers.Dense(units=self.nneuron_, activation=self.activation_, 
                                                use_bias=True, input_shape=self.input_shape_,
                                                kernel_initializer = self.kernel_initializer_,
                                                bias_initializer = self.bias_initializer_,
                                                kernel_regularizer = self.kernel_regularizer_,
                                                bias_regularizer = self.bias_regularizer_, dtype='float32')
        else:
            self.layer_ = tf.keras.layers.Dense(units=self.nneuron_, activation=self.activation_, 
                                                use_bias=True, kernel_initializer = self.kernel_initializer_,
                                                bias_initializer = self.bias_initializer_,
                                                kernel_regularizer = self.kernel_regularizer_,
                                                bias_regularizer = self.bias_regularizer_, dtype='float32')                        
            
    # def construct_layer_manual(self):
    
    #     self.Wij_ = tf.self
            

    def factiv(self, x):

        if (self.activation_ == 'tanh'):
            return tf.math.tanh(x)
        elif (self.activation_ == None):
            return x
        else:
            print('Non available activation function')    

        
        

    # function to return weights and bias
    # function to return the output
    # function to return the input
            