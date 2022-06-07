# %%
import matplotlib.pyplot as plt
#%matplotlib inline

import os
import sys

import numpy as np 
import pandas as pd 
import math

from sklearn.preprocessing import PowerTransformer, StandardScaler

from tensorflow.keras import optimizers
# from keras import optimizers
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Dropout, Input, LeakyReLU, Concatenate, CuDNNLSTM

from libs.functions import NMAE_metric, MARE

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
comms_path = r'/Volumes/SAMSUNG/Backup C/Documentos/Dataset/telecom-sms,call,internet - per_cell'

# %%
# BUILDING OF SQUARE_ID MATRIX
square_id=np.arange(1 , 10001, 1).reshape(100,100) 
square_id = np.flipud(square_id)

# %%
# split a univariate sequence into samples
def split_sequence(sequence, n_steps):
	X, y = list(), list()
	for i in range(len(sequence)):
		# find the end of this pattern
		end_ix = i + n_steps
		# check if we are beyond the sequence
		if end_ix > len(sequence)-1:
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
		X.append(seq_x)
		y.append(seq_y)
	return np.array(X), np.array(y)

# %%
# Evaluate an LSTM model
def evaluate_lstm_model(X):

    dt = X.index.map(lambda a: pd.datetime.fromtimestamp(a/1000))

    df_final = pd.DataFrame({'y':X, 'ds':dt})

    df_final = df_final.reset_index(drop=True)

    # prepare training and test dataset
    train_size = int(len(df_final) * 0.66)
    n_steps=144

    #X, y = split_sequence(np.array(df_final['y']), n_steps)
    X = np.array(df_final['y'][:-1]).reshape(-1, 1)
    y = np.array(df_final['y'].shift(-1)[:-1])

    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # Scaler
    scaler = StandardScaler()

    scaled_train = scaler.fit_transform(X_train)

    #scaled_train = scaled_train.reshape((scaled_train.shape[0], scaled_train.shape[1], 1))

    # Model construction
    model = Sequential()

    model.add(Input(shape=(n_steps, 1), name='Input_y'))
    model.add(LSTM(128, return_sequences=True, name='LSTM1'))
    model.add(Dropout(0.1, name='Dropout1'))
    model.add(LSTM(128, return_sequences=True, name='LSTM2'))
    model.add(Dropout(0.1, name='Dropout2'))
    model.add(Dense(1, name='Output'))

    # Model compiling 
    opt = optimizers.Adamax(learning_rate=0.001)

    model.compile(loss='mean_squared_error', optimizer=opt)

    # Training
    epochs = 100

    model.fit(scaled_train
            , y_train
            , epochs=epochs
            , batch_size=100
            , verbose=2
            )

    # make predictions
    scaled_test = scaler.transform(X_test)

    predictions = model.predict(scaled_test.reshape(scaled_test.shape[0], scaled_test.shape[1], 1))

    forecast = [predictions[i][0][0] for i in range(0, len(predictions))]

    # Plot lines
    plt.figure(figsize=(25,20))
    plt.plot(y_test, label = "Y")
    plt.plot(forecast, label = "Yhat")
    plt.legend()
    plt.show()

    # calculate out of sample error
    error = mean_absolute_percentage_error(y_test, forecast)

    return error

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
# MAIN
error_list = [] 

for cell_id in ids_to_use:
    print(f'\nGET CELL {cell_id} DATA \n')

    y = pd.read_csv(os.path.join(comms_path, f'{cell_id}.csv'), index_col=0)[f'internet_traffic_{cell_id}']
    y.name = 'y'

    #y = np.array(y).reshape(-1,1)

    warnings.filterwarnings("ignore")
    
    err = evaluate_lstm_model(y)

    error_list.append(err)
    
    print(f'\nEND OF CELL {cell_id}\n')
    print('Error: ', str(err))
    print('*'*10)
    print('\n\n')

# %%
print(error_list)
print(np.array(error_list).mean())
# %%

# %%
