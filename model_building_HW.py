# %%
import matplotlib.pyplot as plt
#%matplotlib inline

import os
import sys

import numpy as np 
import pandas as pd 
import math

# Seasonality decomposition
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.seasonal import seasonal_decompose # holt winters 

# single exponential smoothing
from statsmodels.tsa.holtwinters import SimpleExpSmoothing   # double and triple exponential smoothing
from statsmodels.tsa.holtwinters import ExponentialSmoothing

import warnings

from tqdm import tqdm 

# %%
def NMAE(y_true, y_pred):
    return np.mean(np.abs(y_pred - y_true)/np.mean(y_true))

# %%
# STANDARD VARIABLES
comms_path = r'C:\\Users\\patri\\Documents\\Dataset\\telecom-sms,call,internet - per_cell\\'

# %%
# BUILDING OF SQUARE_ID MATRIX
square_id=np.arange(1 , 10001, 1).reshape(100,100) 
square_id = np.flipud(square_id)

# %%
# CHOOSING IDS TO WORK WITH
desired_numbers = 17

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
#plt.savefig(r'C:\\Users\\patri\\Documents\\Github\\milan-telecom-analysis\\results\\check_selected_ids.jpg')

# %%
# MAIN
y_add_nmae=[] 
y_mul_nmae=[]

for cell_id in tqdm(ids_to_use):
    y = pd.read_csv(os.path.join(comms_path, f'{cell_id}.csv'), index_col=0)[f'internet_traffic_{cell_id}']
    y.name = 'y'

    # Set the frequency of the date time index as Monthly start as indicated by the data
    m = 144
    
    alpha = 1/(2*m)

    # MODEL CONSTRUCTION ###################################################################

    warnings.filterwarnings("ignore")
    
    y_add = ExponentialSmoothing(y,trend='add', seasonal_periods=m).fit().fittedvalues

    y_mul = ExponentialSmoothing(y,trend='mul', seasonal_periods=m).fit().fittedvalues

    df_res = pd.DataFrame({'y':y, 'Additive trend':y_add, 'Multiplicative trend':y_mul})
    
    # PLOTTING
    plt.figure(figsize=[23,10])
    df_res[-150:].plot(figsize=[23,18], style=['-', '--', '-.'], linewidth=4)
    plt.legend(loc='best', prop={'size': 25})
    plt.xticks([])
    plt.yticks([])

    # CALC
    y_add_nmae.append(NMAE(y, y_add))
    y_mul_nmae.append(NMAE(y, y_mul))

# %%
print(f'y_add mean = {np.mean(y_add_nmae)}')
print(f'y_mul mean = {np.mean(y_mul_nmae)}')

# %%
