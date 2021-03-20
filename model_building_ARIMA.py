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

from statsmodels.tsa.arima.model import ARIMA
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
# Evaluate an ARIMA model for a given order (p,d,q)
def evaluate_arima_model(X, arima_order):
	# prepare training dataset
	train_size = int(len(X) * 0.66)
	train, test = X[0:train_size], X[train_size:]
	history = [x for x in train]
	# make predictions
	predictions = list()
	for t in tqdm(range(len(test))):
		model = ARIMA(history, order=arima_order)
		model_fit = model.fit()
		yhat = model_fit.forecast()[0]
		predictions.append(yhat)
		history.append(test[t])
	# calculate out of sample error
	error = mean_absolute_percentage_error(test, predictions)
	return error

# %%
# Evaluate combinations of p, d and q values for an ARIMA model
def evaluate_models(dataset, p_values, d_values, q_values):
	dataset = dataset.astype('float32')
	best_score, best_cfg = float("inf"), None
	for p in p_values:
		for d in d_values:
			for q in q_values:
				order = (p,d,q)
				try:
					mse = evaluate_arima_model(dataset, order)
					if mse < best_score:
						best_score, best_cfg = mse, order
					print('ARIMA%s MAPE=%.3f' % (order,mse))
				except:
					continue
	print('Best ARIMA%s MSE=%.3f' % (best_cfg, best_score))

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

    # MODEL CONSTRUCTION ###################################################################
    p_values = [1,2,4,6,8,10]
    d_values=[0,1,2]
    q_values=[0,1,2]

    warnings.filterwarnings("ignore")
    evaluate_models(y, p_values, d_values, q_values)
    
    print(f'\nEND OF CELL {cell_id}\n')
    print('*'*10)
    print('\n\n')
# %%
