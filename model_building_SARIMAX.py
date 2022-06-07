# %%
import matplotlib.pyplot as plt
#%matplotlib inline

import os
import sys

import numpy as np 
import pandas as pd 

sys.path.append(r'C:\\Users\\patri\\Documents\\Github\\milan-telecom-analysis\\libs')
from libs.functions import NMAE_metric

from fbprophet import Prophet
import warnings

from tqdm import tqdm 

from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.api as sm

# %%
def mean_absolute_percentage_error(y_true, y_pred): 

    ## Note: does not handle mix 1d representation
    #if _is_1d(y_true): 
    #    y_true, y_pred = _check_1d_array(y_true, y_pred)

    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# %%
# STANDARD VARIABLES
comms_path = r'/Volumes/SAMSUNG/Backup C/Documentos/Dataset Milano/telecom-sms,call,internet - per_cell'
transport_path = r'transport_modelling public_transport_locations.csv'

# %%
# BUILDING OF SQUARE_ID MATRIX
square_id=np.arange(1 , 10001, 1).reshape(100,100) 
square_id = np.flipud(square_id)

# %%
# CHOOSING IDS TO WORK WITH
desired_numbers = 15

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

ids_to_use = [5738, 5160, 5161, 5060, 5061, 4861, 4761, 4360, 4259, 4359, 4350, 4351, 4352, 4353, 4452, 4453, 4454, 4455, 4456, 4556, 4456, 4356, 4355, 4354, 4250, 4251, 4252, 4253, 4254, 4255, 4256, 4156, 4155, 4154, 4153, 4151]

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
# Evaluate an SARIMAX model
def evaluate_sarimax_model(X, order=(1,1,1), seasonal_order=(1,1,1,6*24*7)):

    dt = X.index.map(lambda a: pd.datetime.fromtimestamp(a/1000))

    df_final = pd.DataFrame({'y':X, 'ds':dt})

    df_final = df_final.set_index('ds')

    # prepare training dataset
    train_size = int(len(df_final) * 0.66)
    # train, test = df_final[0:train_size], df_final[train_size:]


    model = sm.tsa.statespace.SARIMAX(df_final['y'], 
                                      order=order, 
                                      seasonal_order=seasonal_order)
    model_fit = model.fit()

    # make predictions
    predictions = model_fit.predict(start = train_size, end = len(df_final))

    # Plot lines
    plt.figure(figsize=(25,20))
    plt.plot(df_final['ds'], df_final['y'], label = "Y")
    plt.plot(df_final[train_size:], predictions, label = "Yhat")
    plt.legend()
    plt.show()

    # calculate out of sample error
    error = mean_absolute_percentage_error(df_final['y'], predictions[-len(test):])

    return error

# %%
for cell_id in ids_to_use:
    y = pd.read_csv(os.path.join(comms_path, f'{cell_id}.csv'), index_col=0)[f'internet_traffic_{cell_id}']

    dt = y.index.map(lambda a: pd.datetime.fromtimestamp(a/1000))

    df_final = pd.DataFrame({'y':y, 'ds':dt})

    df_final = df_final.reset_index(drop=True)

    df_final = df_final.set_index('ds')

    decompose_data = seasonal_decompose(df_final, model="additive", freq=6*24*7)

    fig, (ax1,ax2,ax3, ax4) = plt.subplots(4,1, figsize=(15,8))
    decompose_data.observed.plot(ax=ax1)
    decompose_data.trend.plot(ax=ax2)
    decompose_data.resid.plot(ax=ax3)
    decompose_data.seasonal.plot(ax=ax4)

# %%
# MAIN
error_list = [] 

for cell_id in ids_to_use:
    print(f'\nGET CELL {cell_id} DATA \n')

    y = pd.read_csv(os.path.join(comms_path, f'{cell_id}.csv'), index_col=0)[f'internet_traffic_{cell_id}']
    y.name = 'y'

    #y = np.array(y).reshape(-1,1)

    warnings.filterwarnings("ignore")
    
    err = evaluate_sarimax_model(y)

    error_list.append(err)
    
    print(f'\nEND OF CELL {cell_id}\n')
    print('*'*10)
    print('\n\n')

# %%
print(error_list)
print(np.array(error_list).mean())
# %%
