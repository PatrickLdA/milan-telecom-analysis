# %%
"""
This code is used to take the original dataframe, that is in a format "one file per day" to a format "one region per day". 
"""
import matplotlib.pyplot as plt
#%matplotlib inline

import os
import sys

import numpy as np 
import pandas as pd 

sys.path.append(r'C:\\Users\\patri\\Documents\\Github\\milan-telecom-analysis\\libs')
import get_milano

# %%
# Paths to read the data and write adapted dataset
in_path = r'C:\\Users\\patri\\Documents\\Dataset\\telecom-sms,call,internet\\'
out_path = r'C:\\Users\\patri\\Documents\\Dataset\\telecom-sms,call,internet - per_cell\\'

ids = np.arange(1, 10001, 1) # Total number of regions

# %%
# The files will be read and all the data, in group of 20 regions, will be transformed to the new format and saved
for split in np.split(ids, 20, axis=0):
    print (f'FROM {split[0]} TO {split[-1]}\n')

    # Reading data from 20 regions
    df = get_milano.comms_selection(comms_directory=in_path, square_id=split) 

    # From every region in the group of 20
    for id in split:
        filtered_cols = [col for col in df.columns if col.endswith(f'_{id}')]
        df[filtered_cols].to_csv(os.path.join(out_path, str(id) + '.csv')) 

    print ('WORK DONE')
    print ('*'*10 + '\n\n')
# %%
