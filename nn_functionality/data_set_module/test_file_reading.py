import numpy as np
import file_module
import data_set as data_set_module

ndim = 5
ncm = 2

input_data = data_set_module.data_set(ndim,ncm,"test_file.txt")


input_data.separate_input_to_target()

xinput = input_data.input_matrix_
ytarget = input_data.target_matrix_

print(input_data.ndata_)
print(input_data.ndim_)
print(input_data.ncm_)

print(xinput)
print(ytarget)

input_data.data_matrix_shuffle()

xinput = input_data.input_matrix_
ytarget = input_data.target_matrix_

print(xinput)
print(ytarget)

xtrain, xval = input_data.data_matrix_split(0.6)

print(xtrain)
print(xval)

input_data2 = data_set_module.data_set(ndim,ncm)

input_data2.input_matrix_ = input_data.input_matrix_
input_data2.target_matrix_ = input_data.target_matrix_

xinput = input_data2.input_matrix_
ytarget = input_data2.target_matrix_

print(xinput)
print(ytarget)