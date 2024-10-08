# .......................................................... #
# Functionality for explainability analysis
# Based on the SHAP library
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
# Wrapper class for SHAP interpretability
# .......................................................... #
#
class model_explainer:
    def __init__(self, model = None, dataset = None, feature_names = None):

        # model: the black box model
        # dataset: a numpy array as a dataset
        # feature_names: a list with string entries as the names of the features
        self.model_ = model
        self.dataset_np_ = dataset
        self.feature_names_ = feature_names

        self.create_pandas_dataset()

        self.explainer_ = shap.KernelExplainer(model, self.dataset_pd_)

    model_ = None
    dataset_np_ = None
    dataset_pd_ = None
    feature_names_ = None
    explainer_ = None
    shap_values_ = None

    
    def create_pandas_dataset(self):

        # Creates the dataset in pandas format
        # Firstly, create a dictionary

        nfeat = len(self.feature_names_)

        lst = []

        for ifeat in range(0,nfeat):

            lst.append( (self.feature_names_[ifeat], self.dataset_np_[:,ifeat]) )


        data_dct = dict(lst)

        # Then, create the pandas dataset using the created dictionary

        self.dataset_pd_ = pd.DataFrame(data=data_dct)

    def compute_shap_values(self):

        self.shap_values_ = self.explainer_.shap_values(self.dataset_pd_)
    
    def create_plots(self, ncm):

         shap.summary_plot(self.shap_values_, self.dataset_pd_)   

         for icm in range(0, ncm):
             shap.summary_plot(self.shap_values_[icm], self.dataset_pd_)

    
