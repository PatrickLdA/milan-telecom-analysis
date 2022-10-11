# %%
import matplotlib.pyplot as plt
#%matplotlib inline

import os
import sys

import numpy as np 
import pandas as pd 

#sys.path.append(r'C:\\Users\\patri\\Documents\\Github\\milan-telecom-analysis\\libs')
from libs.functions import NMAE_metric

from sklearn.preprocessing import PowerTransformer
from sklearn.metrics import mean_squared_error

import tensorflow as tf

from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Input, LeakyReLU
from keras.metrics import MeanSquaredError
from keras.utils import plot_model
import keras

# %%
# STANDARD VARIABLES
tf.config.threading.set_inter_op_parallelism_threads(0)

comms_path = r'Dataset/telecom-sms,call,internet - per_cell'

# %%
# BUILDING OF SQUARE_ID MATRIX
square_id=np.arange(1 , 10001, 1).reshape(100,100) 
square_id = np.flipud(square_id)

# %%
# FUCTION TO RETURN THE LIST OF NEIGHBORS BASED ON RADIUS AND SQUARE_ID
def neighborhood(square_id, id, radius):
    coordinates = np.where(square_id==id) # Center of circle

    # Return the neighbors values
    neighbor = np.array([[square_id[j][i] if  i >= 0 and i < len(square_id) and j >= 0 and j < len(square_id[0]) else np.nan
                for j in range(int(coordinates[0])-radius, int(coordinates[0])+radius+1)]
                    for i in range(int(coordinates[1])-radius, int(coordinates[1])+radius+1)])

    # To_list
    neighbor = np.reshape(neighbor, neighbor.shape[0]*neighbor.shape[1])

    # Deleting the own id from the list
    neighbor = np.delete(neighbor, np.where(neighbor==id))

    # Drop NaN values
    string_list=[]
    [string_list.append(str(int(value))) for value in neighbor if not np.isnan(value)]

    return string_list

# %%
# GET NEIGHBORHOOD TRAFFIC DATA
def prepare_df(radius_max, cell_id, square_id):
    df_output = np.NaN
    to_list = []

    for radius in range (1, radius_max+1):
        to_list = np.append(to_list,neighborhood(square_id, cell_id, radius))

    for cells in to_list:
        data_cell = pd.read_csv(os.path.join(comms_path, f'{cells}.csv'), index_col=0)[f'internet_traffic_{cells}']

        if type(df_output)==type(np.NaN):
            df_output = data_cell.copy()

        else:
            df_output = pd.concat([df_output, data_cell], axis=1)

    return df_output

# %%
neighorrs = 5
cell_id = 6238

if neighorrs==0:
    y = pd.read_csv(os.path.join(comms_path, f'{cell_id}.csv'), index_col=0)[f'internet_traffic_{cell_id}']
    y.name = 'y'

    y_sh = y.shift(periods=-1)
    y_sh.name = 'y_sh'

    df_final = pd.concat([y, y_sh], axis=1)
    df_final = df_final.dropna()

else:
    x = prepare_df(radius_max=neighorrs, cell_id=cell_id, square_id=square_id)

    y = pd.read_csv(os.path.join(comms_path, f'{cell_id}.csv'), index_col=0)[f'internet_traffic_{cell_id}']

    df_final = pd.concat([x, y], axis=1)
    df_final = df_final.rename(columns={f'internet_traffic_{cell_id}': 'y'})
    df_final['y_sh'] = df_final['y'].shift(periods=-1)
    df_final = df_final.dropna()

# SPLITTING AND PREPARING DATASET
size = len(df_final)

X_train = df_final.drop(['y_sh'], axis=1)[:int(0.8*size)]
y_train = df_final['y_sh'][:int(0.8*size)]


X_test = df_final.drop(['y_sh'], axis=1)[int(0.8*size):]
y_test = df_final['y_sh'][int(0.8*size):]


scaler = PowerTransformer()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

y_train = np.array(y_train)
y_test = np.array(y_test)


n_features = X_train.shape[1]  

# PREPARING DATA TO INPUT INTO NNET
X_train = X_train.reshape(-1, 1, n_features)
X_test = X_test.reshape(-1, 1, n_features)


# %%
file_path = f'C:\\Users\\patri\\Documents\\Github\\milan-telecom-analysis\\results\\model_results\\{neighorrs}_neighbors_id_{cell_id}.h5'
network = keras.models.load_model(file_path, custom_objects={'NMAE_metric': NMAE_metric})

# %%
y_predict = network.predict(X_test)
y_predict = [y_predict[i][0][0] for i in range(0, len(y_predict))]

# %%
# PLOTTING TEST
x_plot = np.arange(0, len(y_predict), 1)

plt.figure(figsize=(20, 14))
plt.plot(x_plot, y_test, label='test')
plt.plot(x_plot, y_predict, label='predict')

plt.legend(loc='best')

plt.savefig(f'C:\\Users\\patri\\Documents\\Github\\milan-telecom-analysis\\results\\{cell_id}_id_{neighorrs}_neigh_predict.png')

plt.show()

# %%
