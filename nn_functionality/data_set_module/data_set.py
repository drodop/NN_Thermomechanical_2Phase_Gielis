# .......................................................... #
# Functionality for dataset processing
# D. Rodopoulos
# .......................................................... #
# ========================================================== #
# Basic modules
# ========================================================== #
import math
import numpy as np

# ========================================================== #
# Imports of custom modules
import file_module

#   
# .......................................................... #
# Dataset shuffle
# .......................................................... #
#
def data_shuffle(xarray):

    np.random.shuffle(xarray)

    return xarray

#   
# .......................................................... #
# Dataset split
# .......................................................... #
#
def split_dataset(xi,yi,split_percentage):
    
    # Splits the dataset into the training dataset and validation dataset by a given percentage
    xdata = np.column_stack((xi,yi))
    xdata = data_shuffle(xdata) # Shuffle the data
    
    nrow = xdata.shape[0]
    split_position = math.ceil(nrow*split_percentage)
    
    xtrain = xdata[:split_position]
    xval  = xdata[split_position:]

    return xtrain, xval

#   
# .......................................................... #
# Dataset normalize
# .......................................................... #
#
def data_normalize(xdata):
    
    # Find the mean value of the dataset. Caution: It finds the mean for each column.
    # Be carefull of the form of the array! For a nxm, the mean value is an 1d array with size = m
    mean_value = xdata.mean(axis=0)
    xdata = xdata - mean_value
    
    standard_dev = xdata.std(axis=0)
    xdata = xdata / standard_dev
    
    return xdata

#   
# .......................................................... #
# Linear transformation of vector data
# .......................................................... #
#
def data_linear_transform(xarray, ub1, ub2):
    
    # CAUTION: The dataset should be ordered!
    # xarray: Array to be linearly transformed
    # ub1: The requested low bound
    # ub1: The requested upper bound
    # xnew: Transformed array to be returned

    xb1 = xarray[0]
    xb2 = xarray[-1]

    lhmat = np.array([[xb1,1.0],[xb2,1.0]])
    rhmat = np.array([[ub1],[ub2]])

    scale_coefs = np.linalg.solve(lhmat,rhmat)

    jac = scale_coefs[0][0]

    #print(scale_coefs)
    
    xnew = scale_coefs[0][0]*xarray + scale_coefs[1][0]

    return xnew, jac

# ========================================================== #
# Class data_set
# ========================================================== #
class data_set:

    def __init__(self, ndim, ncm, inp_file = None):
        
        if inp_file is not None:
            self.input_file_ = inp_file

        self.ndim_ = ndim
        self.ncm_ = ncm
     
    input_file_ = None
    ndim_ = 2
    ncm_ = 1
    ndata_ = 0
    input_matrix_ = None
    target_matrix_ = None 

    def read_file(self):
        file_list = file_module.readDataFile_general(self.input_file_, self.ndim_, self.ncm_)
        return np.array(file_list)
    
    def separate_input_to_target(self):

        data_array = self.read_file()

        nrow = data_array.shape[0]

        self.ndata_ = nrow

        xinput  = np.zeros((nrow, self.ndim_))
        ytarget = np.zeros((nrow, self.ncm_))

        for irow in range(0, nrow):
            for idim in range(0, self.ndim_):
                xinput[irow][idim] = data_array[irow][idim]

            iref = self.ndim_
            for icm in range(0, self.ncm_):
                ytarget[irow][icm] = data_array[irow][iref + icm]   

        self.input_matrix_ = xinput
        self.target_matrix_ = ytarget

    def data_matrix_shuffle(self):
   
        xinput = self.input_matrix_
        ytarget = self.target_matrix_   

        xdata = np.column_stack((xinput, ytarget))
        xdata = data_shuffle(xdata) # Shuffle the data    

        xinput = np.reshape(xdata[:,0:self.ndim_], (self.ndata_,self.ndim_))
        ytarget = np.reshape(xdata[:,self.ndim_:self.ndim_+self.ncm_], (self.ndata_,self.ncm_))

        self.input_matrix_ = xinput
        self.target_matrix_ = ytarget

    def data_matrix_split(self,split_percentage):
   
        xinput = self.input_matrix_
        ytarget = self.target_matrix_   

        xtrain, xval = split_dataset(xinput, ytarget, split_percentage)  

        return xtrain, xval

    def data_matrix_normalize(self):
   
        xinput = self.input_matrix_
        ytarget = self.target_matrix_   

        xdata = np.column_stack((xinput, ytarget))

        xdata = data_normalize(xdata)

        xinput = np.reshape(xdata[:,0:self.ndim_], (self.ndata_,self.ndim_))
        ytarget = np.reshape(xdata[:,self.ndim_:self.ndim_+self.ncm_], (self.ndata_,self.ncm_))

        self.input_matrix_ = xinput
        self.target_matrix_ = ytarget
