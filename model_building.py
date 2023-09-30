# %%
import matplotlib.pyplot as plt
#%matplotlib inline

import os
import sys

import numpy as np 
import pandas as pd 
import math

from libs.functions import NMAE_metric, MARE

from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_classif

from sklearn.preprocessing import PowerTransformer, StandardScaler
from sklearn.metrics import mean_squared_error

import tensorflow as tf

from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Dropout, Input, LeakyReLU, Concatenate
from keras.metrics import MeanSquaredError, MeanAbsolutePercentageError
# from keras.utils import plot_model
from keras.utils.vis_utils import plot_model
from keras import optimizers

import time 

import pickle

# %%
# STANDARD VARIABLES
tf.config.threading.set_inter_op_parallelism_threads(4)


comms_path = 'Dataset/telecom-sms,call,internet - per_cell'
transport_path = 'transport_modelling/public_transport_locations.csv'

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
    to_drop = [ids for ids in used_ids if (ids in traffic.columns) or (ids in ['internet_traffic_' + str(cell_id)])]
    to_drop.append(f'internet_traffic_{cell_id}')
    traffic = traffic.drop(to_drop, axis=1, errors='ignore')

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
    
    for transport in traffic.drop(['y'], axis=1).columns:
        id_traffic = np.where(square_id==int(transport[17:]))
    
        distance = math.sqrt((y_locs[0]-id_traffic[0])**2 + (y_locs[1]-id_traffic[1])**2)
    
        if distance < n_features:
            cols.append(transport)
    
    df_output = traffic[cols]
    df_output.drop([f'internet_traffic_{cell_id}'], axis=1, errors='ignore')

    print('Used cols:')
    print(cols)

    return df_output

# %%
# CHOOSING IDS TO WORK WITH
# As the dataset have 10000 regions, some poins are choosen with a regular distance
desired_numbers = 18

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

# IDs of event and core tests
#ids_to_use = [5738, 5160, 5161, 5060, 5061, 4861, 4761, 4360, 4259, 4359, 4350, 4351, 4352, 4353, 4452, 4453, 4454, 4455, 4456,
#              4556, 4456, 4356, 4355, 4354, 4250, 4251, 4252, 4253, 4254, 4255, 4256, 4156, 4155, 4154, 4153, 4151]
#ids_to_use = [607, 8169, 5738]

# Continue execution
ids_to_use=ids_to_use[22:]

# %%
# Check selected ids
ids_to_print = []
matrix_print = np.zeros([100,100])

for values in ids_to_use:
    matrix_print[np.where(square_id==values)] = 1

plt.figure(figsize=[20,20])
plt.imshow(matrix_print)
plt.colorbar()
plt.savefig('check_selected_ids.jpg')

# %%
# MAIN PROCESS
"""
The models will be constructed with the following variations:
- Transport hubs processing: True, False
- Neighborhoods: from 1 to 6
- All the regions choosen before
"""

data_frame_results = np.NaN 

for transport_hubs in [True, False]: # Transport hubs processing
    for neighorrs in range(5,6): # Number of neighborhoods considered
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

                    df_final = pd.concat([x, y, transport_ids_used], axis=1)

                else:
                    df_final = pd.concat([x, y], axis=1)

                df_final = df_final.rename(columns={f'internet_traffic_{cell_id}': 'y'})

                # Shifting the y time series to make a predictive model
                df_final['y_sh'] = df_final['y'].shift(periods=-1)

                df_final = df_final.dropna()

            # SPLITTING AND PREPARING DATASET
            size = len(df_final)

            X_train_y = df_final['y'][:int(0.8*size)] # y time series
            X_train_other = df_final.drop(['y_sh', 'y'], axis=1)[:int(0.8*size)] # Other (neighborhoods + transport_hubs)
            y_train = df_final['y_sh'][:int(0.8*size)] # Target variable y shifted


            X_test_y = df_final['y'][int(0.8*size):] # y time series
            X_test_other = df_final.drop(['y_sh', 'y'], axis=1)[int(0.8*size):] # Other (neighborhoods + transport_hubs)
            y_test = df_final['y_sh'][int(0.8*size):] # Target variable y shifted


            scaler_y = StandardScaler()  # y time series scaler
            scaler_other = StandardScaler() # other series scaler

            X_train_y = np.array(X_train_y).reshape(-1, 1)
            X_train_y = scaler_y.fit_transform(X_train_y)

            X_train_other = scaler_other.fit_transform(X_train_other)

            X_test_y = np.array(X_test_y).reshape(-1, 1)
            X_test_y = scaler_y.transform(X_test_y)

            X_test_other = scaler_other.transform(X_test_other)

            y_train = np.array(y_train)
            y_test = np.array(y_test)


            n_features_y = X_train_y.shape[1]  
            n_features_other = X_train_other.shape[1] 
            
            # PREPARING DATA TO INPUT INTO NNET
            X_train_y = X_train_y.reshape(-1, 1, n_features_y)
            X_train_other = X_train_other.reshape(-1,1,n_features_other)

            X_test_y = X_test_y.reshape(-1, 1, n_features_y)
            X_test_other = X_test_other.reshape(-1, 1, n_features_other)
        

            # MODEL CONSTRUCTION ###################################################################
            # Y input
            Input_y = Input(shape=(1, n_features_y), name='Input_y')

            LSTM1 = LSTM(144, activation='relu', return_sequences=True, input_shape=(1, n_features_y), name='LSTM1')(Input_y)

            Dropout1 = Dropout(0.2, name='Dropout1')(LSTM1)

            LSTM2 = LSTM(144, activation='relu', return_sequences=True, input_shape=(1, n_features_y), name='LSTM2')(Dropout1)

            Dropout2 = Dropout(0.2, name='Dropout2')(LSTM2)

            Dense1 = Dense(n_features_y*4, activation='relu', name='Dense1')(Dropout2)

            Dropout3 = Dropout(0.2, name='Dropout3')(Dense1)

            Dense2 = Dense(n_features_y*2, activation='relu', name='Dense2')(Dropout3)
            ################

            # Other cells input
            Input_other = Input(shape=(1, n_features_other), name='Input_other')

            Encoding_other_1 = Dense(int(n_features_other), activation='relu', name='Encoding_other_1')(Input_other)

            Encoding_other_2 = Dense(int(n_features_other/2), activation='relu', name='Encoding_other_2')(Encoding_other_1)

            LSTM_other_1 = LSTM(144, activation='relu', return_sequences=True, input_shape=(1, n_features_other), name='LSTM_other_1')(Encoding_other_1)
            # LSTM_other_1 = LSTM(144, activation='relu', return_sequences=True, input_shape=(1, n_features_other), name='LSTM_other_1')(Encoding_other_2) # Test with two consecutive encondings

            Dropout_other_1 = Dropout(0.2, name='Dropout_other_1')(LSTM_other_1)

            LSTM_other_2 = LSTM(144, activation='relu', return_sequences=True, input_shape=(1, n_features_other), name='LSTM_other_2')(Dropout_other_1)

            Decoder_other_1 = Dense(n_features_other*2, activation='relu', name='Decoder_other_1')(LSTM_other_2)

            Dropout_other_2 = Dropout(0.4, name='Dropout_other_2')(Decoder_other_1)

            Dense_other_2 = Dense(n_features_other*4, activation='relu', name='Dense_other_2')(Dropout_other_2)

            Dropout_other_3 = Dropout(0.4, name='Dropout_other_3')(Dense_other_2)

            Dense_other_3 = Dense(n_features_other*2, activation='relu', name='Dense_other_3')(Dropout_other_3)
            ##############

            concat_layer = Concatenate()([Dense2, Dense_other_3])
            
            Output = Dense(1, name='Output')(concat_layer)

            network = Model(inputs=[Input_y, Input_other], outputs=Output,)
            ######################################################################################

            opt = tf.keras.optimizers.Adamax(learning_rate=0.001)
            network.compile(loss='mean_squared_error', optimizer=opt, metrics=[NMAE_metric, MARE])

            plot_model(network, to_file='network.png', show_shapes=True,)

            epochs = 50

            beg = time.time()

            model = network.fit({'Input_y':X_train_y, 'Input_other':X_train_other}
                                , y_train
                                , epochs=epochs, batch_size=100
                                , verbose=2
                                )
            
            tot.append(time.time() - beg )

            print(f'\nEND OF CELL {cell_id}, NEIGHBORHOOD {neighorrs}, HUBS {transport_hubs}\n')
            print('*'*10)
            print('\n\n')

            if type(data_frame_results) == type(np.NaN):
                data_frame_results = pd.DataFrame(model.history['NMAE_metric'], columns=[f'{cell_id}_nmae'])
                mare = pd.DataFrame(model.history['MARE'], columns=[f'{cell_id}_mare'])

            else:
                data_frame_results[f'{cell_id}_nmae'] = model.history['NMAE_metric']
                mare[f'{cell_id}_mare'] = model.history['MARE']

            ###
            if(transport_hubs):
                network.save(filepath=f'results/hourly/model_results/{neighorrs}_neighbors_id_{cell_id}_feature_selec.h5')

            else:
                network.save(filepath=f'results/hourly/model_results/{neighorrs}_neighbors_id_{cell_id}.h5')


            y_predict = network.predict({'Input_y':X_test_y, 'Input_other':X_test_other})
            y_predict = [y_predict[i][0][0] for i in range(0, len(y_predict))]

            # CSV saves
            y_yhat_dict = {'y':y_test, 'y_hat':y_predict}
            df_real_and_predicts = pd.DataFrame(y_yhat_dict)

            if transport_hubs:
                df_real_and_predicts.to_csv(f'results/hourly/model_csvs/{cell_id}_id_{neighorrs}_neigh_predict_feature_selec.csv')
            else:
                df_real_and_predicts.to_csv(f'results/hourly/model_csvs/{cell_id}_id_{neighorrs}_neigh_predict.csv')

            # PLOTTING TEST
            x_plot = np.arange(0, len(y_predict), 1)

            plt.figure(figsize=(20, 14))
            plt.plot(x_plot, y_test, label='test')
            plt.plot(x_plot, y_predict, label='predict')

            plt.legend(loc='best')

            
            if transport_hubs:
                plt.savefig(f'results/hourly/model_plots/{cell_id}_id_{neighorrs}_neigh_predict_feature_selec.png')

            else:
                plt.savefig(f'results/hourly/model_plots/{cell_id}_id_{neighorrs}_neigh_predict.png')

        
        if transport_hubs:
            data_frame_results.to_csv(f'results/hourly/compiled_results/{neighorrs}_neighbors_feature_selec.csv')
            mare.to_csv(f'results/hourly/compiled_results/{neighorrs}_neighbors_feature_selec_MARE.csv')

            with open(f'results/hourly/compile_time/{neighorrs}_time_feature_selec.pickle', 'wb') as timecomp:
                pickle.dump(tot, timecomp)
        
        else:
            data_frame_results.to_csv(f'results/hourly/compiled_results/{neighorrs}_neighbors.csv')
            mare.to_csv(f'results/hourly/compiled_results/{neighorrs}_neighbors_MARE.csv')

            with open(f'results/hourly/compile_time/{neighorrs}_time.pickle', 'wb') as timecomp:
                pickle.dump(tot, timecomp)
# %%
