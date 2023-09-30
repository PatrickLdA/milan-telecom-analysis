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
event_test = [5738, 5160, 5161, 5060, 5061, 4861, 4761, 4360, 4259, 4359, 4350, 4351, 4352, 4353, 4452, 4453, 4454, 4455, 4456,
              4556, 4456, 4356, 4355, 4354, 4250, 4251, 4252, 4253, 4254, 4255, 4256, 4156, 4155, 4154, 4153, 4151]
core_test = [607, 8169, 5738]

# %%
# Check selected ids
ids_to_print = []
matrix_print = np.zeros([100,100])

for values in ids_to_use:
    matrix_print[np.where(square_id==values)] = 1

plt.figure(figsize=[20,20])
plt.imshow(matrix_print)
plt.colorbar()

# %%
matrix_print = np.zeros([100,100])

# Preencher a matriz com valores para cada grupo
for values in ids_to_use:
    matrix_print[np.where(square_id==values)] = 1
for values in event_test:
    matrix_print[np.where(square_id==values)] = 2
for values in core_test:
    matrix_print[np.where(square_id==values)] = 3

# Criar o gráfico de dispersão
plt.figure(figsize=(8, 8))
plt.imshow(matrix_print, cmap='viridis', interpolation='none', origin='lower', extent=[0, 100, 0, 100])

# Adicionar a legenda
legend_labels = {
    1: 'Distributed test',
    2: 'Event test',
    3: 'Core test'
}

# Obter a colormap usada no scatterplot
cmap = plt.get_cmap('viridis')

for label_id, label_text in legend_labels.items():
    colors = cmap(int(label_id)*100)
    plt.scatter([], [], color=colors, marker='o', label=label_text)

# Adicionar a legenda ao gráfico
plt.legend(scatterpoints=1, frameon=False, labelspacing=1, loc='upper left', bbox_to_anchor=(1, 1))


# Mostrar o gráfico
plt.axis('off')

plt.show()
# %%
