# %%
import matplotlib.pyplot as plt
#%matplotlib inline

import os
import sys

import numpy as np 
import pandas as pd 
import math

sys.path.append(r'C:\\Users\\patri\\Documents\\Github\\milan-telecom-analysis\\libs')
from functions import NMAE_metric

from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_classif
from sklearn.metrics import mean_squared_error

from fbprophet import Prophet
import warnings

from tqdm import tqdm 

# %%
def mean_absolute_percentage_error(y_true, y_pred): 

    ## Note: does not handle mix 1d representation
    #if _is_1d(y_true): 
    #    y_true, y_pred = _check_1d_array(y_true, y_pred)

    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# %%
# STANDARD VARIABLES
comms_path = r'C:\\Users\\patri\\Documents\\Dataset\\telecom-sms,call,internet - per_cell\\'
transport_path = r'transport_modelling\\public_transport_locations.csv'

# %%
# BUILDING OF SQUARE_ID MATRIX
square_id=np.arange(1 , 10001, 1).reshape(100,100) 
square_id = np.flipud(square_id)

# %%
# Evaluate an PROPHET model
def evaluate_prophet_model(X):
    X['date'] = pd.to_datetime(X['date'])

    # prepare training dataset
    train_size = int(len(X) * 0.66)
    train, test = X[0:train_size], X[train_size:]


    model = PROPHET()
    model.fit(train)

    # make predictions
    predictions = model.predict(test['date'])

    # calculate out of sample error
    error = mean_absolute_percentage_error(test, predictions)

    return error

# %%
# CHOOSING IDS TO WORK WITH
desired_numbers = 3

matrix_logs = [square_id]
aux_matrix = []
ids_to_use = []

while (len(matrix_logs) < desired_numbers):
    for matrix in matrix_logs:
        horizontal = matrix.shape[0]
        vertical = matrix.shape[1]

        a, b, c, d = matrix[:horizontal//2, :vertical//2], matrix[horizontal//2:, :vertical//2], matrix[:horizontal//2:, vertical//2:], matrix[horizontal//2:, vertical//2:]
        
        aux_matrix.append(a)
        aux_matrix.append(b)
        aux_matrix.append(c)
        aux_matrix.append(d)
    
    matrix_logs = aux_matrix 
    aux_matrix = []

for matrix in matrix_logs:
    ids_to_use.append(matrix[matrix.shape[0]//2][matrix.shape[1]//2])

# %%
# Check selected ids
ids_to_print = []
matrix_print = np.zeros([100,100])

for values in ids_to_use:
    matrix_print[np.where(square_id==values)] = 1

plt.figure(figsize=[20,20])
plt.imshow(matrix_print)
plt.colorbar()
plt.savefig(r'C:\\Users\\patri\\Documents\\Github\\milan-telecom-analysis\\results\\check_selected_ids.jpg')

# %%
# MAIN
data_frame_results = np.NaN 

for cell_id in ids_to_use:
    print(f'\nGET CELL {cell_id} DATA \n')

    y = pd.read_csv(os.path.join(comms_path, f'{cell_id}.csv'), index_col=0)[f'internet_traffic_{cell_id}']
    y.name = 'y'

    y = np.array(y).reshape(-1,1)

    warnings.filterwarnings("ignore")
    evaluate_prophet_model(y)
    
    print(f'\nEND OF CELL {cell_id}\n')
    print('*'*10)
    print('\n\n')
# %%
