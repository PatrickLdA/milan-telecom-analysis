# %%
import matplotlib.pyplot as plt
#%matplotlib inline

import os
import sys

import numpy as np 
import pandas as pd 
import math

sys.path.append(r'C:\\Users\\patri\\Documents\\Github\\milan-telecom-analysis\\libs')
from functions import NMAE_metric, MARE

from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_classif

from sklearn.preprocessing import PowerTransformer, StandardScaler
from sklearn.metrics import mean_squared_error

import tensorflow as tf

from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Dropout, Input, LeakyReLU, Concatenate
from keras.metrics import MeanSquaredError, MeanAbsolutePercentageError
from keras.utils import plot_model
from keras import optimizers

import time 

import pickle

# %%
# STANDARD VARIABLES
tf.config.threading.set_inter_op_parallelism_threads(4)


comms_path = r'C:\\Users\\patri\\Documents\\Dataset\\telecom-sms,call,internet - per_cell\\'
transport_path = r'transport_modelling\\public_transport_locations.csv'

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
def transport_hubs_select(y, n_features, cell_id, used_ids):
    traffic = y
    traffic.name = 'y'

    hubs = pd.read_csv(transport_path, index_col=0)
    hubs = hubs['cell_id'].unique()

    for id in hubs:
        id = int(id)
        x = pd.read_csv(os.path.join(comms_path, f'{id}.csv'), index_col=0)[f'internet_traffic_{id}']
        traffic = pd.concat([traffic, x], axis=1)

    traffic = traffic.fillna(0)

    # Dropping cell_id or used ids from neighborhood
    to_drop = [ids for ids in used_ids if ids in traffic.columns or ids in ['internet_traffic_' + str(cell_id)]]
    traffic.drop(to_drop, axis=1)

    ############### FEATURE SELECTION FUNCTIONS #####################################################
    # SelectKBeast
    #bestfeatures = SelectKBest(k=n_features, score_func=f_regression)
    #bestfeatures.fit(traffic.drop(['y'], axis=1), traffic['y'])
    
    #cols = bestfeatures.get_support(indices=True)
    #df_output = traffic.drop(['y'], axis=1).iloc[:,cols]

    # Correlation
    #correlation = traffic.corr()['y'].abs()
    #
    #cols = correlation.sort_values(ascending=False)[1:n_features+1].index
    #
    #df_output = traffic[cols]

    # Distance based
    cols=[]
    y_locs = np.where(square_id==int(cell_id))
    
    for transport in  traffic.drop(['y'], axis=1).columns:
        id_traffic = np.where(square_id==int(transport[17:]))
    
        distance = math.sqrt((y_locs[0]-id_traffic[0])**2 + (y_locs[1]-id_traffic[1])**2)
    
        if distance < n_features:
            cols.append(transport)
    
    df_output = traffic[cols]

    print('Used cols:')
    print(cols)

    return df_output

# %%
# CHOOSING IDS TO WORK WITH
# As the dataset have 10000 regions, some poins are choosen with a regular distance
desired_numbers = 33

matrix_logs = [square_id] # Original regions matrix
aux_matrix = []
ids_to_use = []

while (len(matrix_logs) < desired_numbers):
    for matrix in matrix_logs:
        horizontal = matrix.shape[0]
        vertical = matrix.shape[1]

        # From the original regions matrix, the regions where divided in 4 subregions
        a, b, c, d = matrix[:horizontal//2, :vertical//2], matrix[horizontal//2:, :vertical//2], matrix[:horizontal//2:, vertical//2:], matrix[horizontal//2:, vertical//2:]
        
        aux_matrix.append(a)
        aux_matrix.append(b)
        aux_matrix.append(c)
        aux_matrix.append(d)
    
    # All the regions (resulted from the partitioning of the previous regions in 4 subregions each) where saved
    matrix_logs = aux_matrix 
    aux_matrix = []

# Afted the partition of the original regions matrix in a desired number of subregions, the central point of every subregion is used
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
# MAIN PROCESS
"""
The models will be constructed with the following variations:
- Transport hubs processing: True, False
- Neighborhoods: from 1 to 6
- All the regions choosen before
"""

data_frame_results = np.NaN 

for transport_hubs in [True]: # Transport hubs processing
    for neighorrs in [5]: # Number of neighborhoods considered
        tot = []

        for cell_id in ids_to_use: # Regions selected
            print(f'\nGET CELL {cell_id} DATA TO {neighorrs} NEIGHBORS\n')

            if neighorrs==0:
            #    y = pd.read_csv(os.path.join(comms_path, f'{cell_id}.csv'), index_col=0)[f'internet_traffic_{cell_id}']
            #    y.name = 'y'
            #
            #    y_sh = y.shift(periods=-1)
            #    y_sh.name = 'y_sh'
            #
            #    df_final = pd.concat([y, y_sh], axis=1)
            #    df_final = df_final.dropna()
                pass

            else:
                # Get the neighborhoods time serie
                x = prepare_df(radius_max=neighorrs, cell_id=cell_id, square_id=square_id)

                # Get the processed region time series
                y = pd.read_csv(os.path.join(comms_path, f'{cell_id}.csv'), index_col=0)[f'internet_traffic_{cell_id}']
                
                # Case the transport hubs are considered
                if transport_hubs:
                    transport_ids_used = transport_hubs_select(y, n_features=20, cell_id=cell_id, used_ids=x.columns)

                    df_final = pd.concat([x, y.shift(periods=-1), transport_ids_used], axis=1)

                else:
                    df_final = pd.concat([x, y], axis=1)

                df_final = df_final.rename(columns={f'internet_traffic_{cell_id}': 'y'})

                # Shifting the y time series to make a predictive model
                df_final['y_sh'] = df_final['y'].shift(periods=-1)

                df_final = df_final.dropna()

            # SPLITTING AND PREPARING DATASET
            size = len(df_final)

            X_train_other = df_final.drop(['y_sh', 'y'], axis=1)[:int(0.8*size)] # Other (neighborhoods + transport_hubs)
            X_test_other = df_final.drop(['y_sh', 'y'], axis=1)[int(0.8*size):] # Other (neighborhoods + transport_hubs)

            y_train = df_final['y_sh'][:int(0.8*size)] # Target variable y shifted

            y_test = df_final['y_sh'][int(0.8*size):] # Target variable y shifted

            scaler_other = StandardScaler() # other series scaler

            X_train_other = scaler_other.fit_transform(X_train_other)

            X_test_other = scaler_other.transform(X_test_other)

            n_features_other = X_train_other.shape[1] 
            
            # PREPARING DATA TO INPUT INTO NNET
            X_train = X_train_other.reshape(-1,1,n_features_other)

            X_test = X_test_other.reshape(-1, 1, n_features_other)
        

            # MODEL CONSTRUCTION ###################################################################
            Input_other = Input(shape=(1, n_features_other), name='Input_other')

            Encoding_other_1 = Dense(int(n_features_other), activation='relu', name='Encoding_other_1')(Input_other)

            Encoding_other_2 = Dense(int(n_features_other/2), activation='relu', name='Encoding_other_2')(Encoding_other_1)
            ##############

            network = Model(inputs=Input_other, outputs=Encoding_other_2,)
            ######################################################################################

            opt = optimizers.Adamax(learning_rate=0.001)
            network.compile(loss='mean_squared_error', optimizer=opt, metrics=[NMAE_metric, MARE])


            epochs = 50

            model = network.fit(X_train
                                , y_train
                                , epochs=epochs, batch_size=100
                                , verbose=2
                                )
    
            print(f'\nEND OF CELL {cell_id}, NEIGHBORHOOD {neighorrs}, HUBS {transport_hubs}\n')
            print('*'*10)
            print('\n\n')