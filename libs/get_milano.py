# %%
# IMPORTING
import pandas as pd 
import os
import numpy as np

# %%
# GET FUNCTIONS
def comms_selection(comms_directory, dates=[""], square_id=np.arange(1 ,10001, 1), time_interval=np.arange(1383260400000, 1388616600001, 600000), columns = ['square_id', 'time_interval', 'country_code', 'sms_in', 'sms_out', 'call_in', 'call_out', 'internet_traffic']):
    """
    Take all the communication data from Milano grid dataset.

    ARGUMENTS
    comms_directory
    - Description: directory you have the dataset.
    dates
    - Description: date you want to get. From '2013-11-01' to '2014-01-01'.
    - Standard: takes all data.
    square_id
    - Description: id of the cell, from '1' to '10000'.
    - Standard: takes all data.
    time_interval
    - Description: time interval of the data. From 1383260400000 to 1388616600001
    - Standard: takes all data.

    RETURN
    - Dataframe with all data requested.
    """
    
    column_names = ['square_id', 'time_interval', 'country_code', 'sms_in', 'sms_out', 'call_in', 'call_out', 'internet_traffic']
    filtered = [column for column in column_names if column in columns]


    i=0
    #days_done = 0 # Not working
    #days_to_do = len(dates) # Not working
    
    # Sufixes building for date filtering
    sufixes=[]
    for date in dates:
        sufixes.append(date+'.txt')

    # Deal with int type pass:
    #if type(square_id) == int:
    #    square_id = [square_id]
    #if type(time_interval) == int:
    #    time_interval = [time_interval]

    # Initial consulting in the dataset files
    for filename in os.listdir(comms_directory):
        #if days_done==days_to_do: # Not working
            #break

        if filename.endswith(tuple(sufixes)):
        # elif filename.endswith(tuple(sufixes)):
            if i==0:
                df = pd.read_csv(comms_directory+filename, sep='\t', header=None, names=column_names, index_col=False)
                
                df = df[filtered]

                df = df[df['square_id'].isin(square_id)] # Id filtering
                df = df[df['time_interval'].isin(time_interval)] # Time filtering
                i+=1
                #days_done+=1 # Not working

            else:
                df_aux = pd.read_csv(comms_directory+filename, sep='\t', header=None, names=column_names, index_col=False)

                df_aux = df_aux[filtered]
                
                df_aux = df_aux[df_aux['square_id'].isin(square_id)] # Id filtering
                df_aux = df_aux[df_aux['time_interval'].isin(time_interval)] # Time filtering

                df = df.append(df_aux)

                #days_done+=1 # Not working
        print(f'Done in {filename}')

    df = df.groupby(['square_id', 'time_interval']).sum().reset_index()

    # Final construction
    df_final=pd.DataFrame(index=df['time_interval'].unique())

    for ids in square_id:
        to_sum = df[df['square_id']==int(float(ids))].set_index('time_interval').drop(['square_id'], axis=1).add_suffix('_'+str(ids))
        
        df_final = pd.concat([df_final, to_sum], axis=1)
    return df_final

def social_pulse():
    """
    NOT READY YET
    """
    df = gpd.read_file(social_directory)
    return df

def meteo(directory, measurements=['Wind Direction', 'Temperature', 'Relative Humidity', 'Wind Speed', 'Precipitation', 'Global Radiation', 'Atmospheric Pressure', 'Net Radiation']):
    """
    Take all the meteorological data from Milano.

    ARGUMENTS
    measurements
    - Description: measurement type of the sensors to be taken.
    directory
    - Description: directory you have the dataset.
    RETURN
    - Dataframe with all data requested.
    """    
    column_sensors = ['id', 'time', 'Measurement']
    
    reference = pd.read_csv(directory+'\\mi_meteo_legend_updated.csv')
    i=0

    for measures in measurements:
        if measures not in reference['Measurement'].unique():
            return 'Measurement(s) not found!'

    desired_sensors = reference[reference['Measurement'].isin(tuple(measurements))]['id'] # Measurements filtering

    desired_sensors_list=[]
    for regs in desired_sensors:
        desired_sensors_list.append(str(regs)+'.csv')

    for filename in os.listdir(directory):
        if filename=='mi_meteo_legend_updated.csv':
            continue

        if filename.endswith(tuple(desired_sensors_list)):
            if i==0:
                df=pd.read_csv(directory+"\\"+filename, header=None, names=column_sensors)
                i+=1

            else:
                df_aux = pd.read_csv(directory+"\\"+filename, header=None, names=column_sensors)
                df = df.append(df_aux)

    df['Variable']=df['id'].apply(lambda x: str(reference[reference['id']==x]['Measurement'].values)[2:-2])
    df['Unit']=df['id'].apply(lambda x: str(reference[reference['id']==x]['Unit'].values)[2:-2])

    # Final construction
    df_final=pd.DataFrame(index=df['time'].unique())

    for ids in df['id'].unique():
        to_sum = df[df['id']==int(ids)].set_index('time').drop(['id'], axis=1).add_suffix('_'+str(ids))
        
        #df_final = df_final.merge(df[df['square_id']==int(ids)].set_index('time_interval').add_suffix('_'+str(ids)), how='outer')
        df_final = pd.concat([df_final, to_sum], axis=1)
    return df_final

def precipitation(path):
    """
    Take all the precipitation data from Milano.

    ARGUMENTS
    path
    - Description: path to the data.

    RETURN
    - Dataframe with all data requested.
    """
    columns=['timestamp', 'id', 'intensity', 'coverage', 'type']
    df = pd.read_csv(path, names=columns)

    df_final=pd.DataFrame(index=df['timestamp'].unique())

    for ids in df['id'].unique():
        to_sum = df[df['id']==int(ids)].set_index('timestamp').drop(['id'], axis=1).add_suffix('_'+str(ids))
        df_final = pd.concat([df_final, to_sum], axis=1)
    return df_final

def mi_to_provinces(directory, dates=[""], square_id=np.arange(1 ,10001, 1), time_interval=np.arange(1383260400000, 1388616600001, 600000)):
    """
    Take all the communication data from Milano grid to another cities.

    ARGUMENTS
    directory
    - Description: directory you have the dataset.
    dates
    - Description: date you want to get. From '2013-11-01' to '2014-01-01'.
    - Standard: takes all data.
    square_id
    - Description: id of the cell, from '1' to '10000'.
    - Standard: takes all data.
    time_interval
    - Description: time interval of the data. From 1383260400000 to 1388616600001
    - Standard: takes all data.

    RETURN
    - Dataframe with all data requested.
    """
    
    column_names = ['square_id', 'province', 'time_interval', 'square_to_province', 'province_to_square']

    i=0
    #days_done = 0 # Not working
    #days_to_do = len(dates) # Not working
    
    # Sufixes building for date filtering
    sufixes=[]
    for date in dates:
        sufixes.append(date+'.txt')

    # Initial consulting in the dataset files
    for filename in os.listdir(directory):
    #    if days_done==days_to_do: # Not working
    #        break # Not working

        if filename.endswith(tuple(sufixes)):
        #elif filename.endswith(tuple(sufixes)): # Not working
            if i==0:
                df = pd.read_csv(directory+"/"+filename, sep='\t', header=None, names=column_names)
    
                df = df[df['square_id'].isin(square_id)] # Id filtering
                df = df[df['time_interval'].isin(time_interval)] # Time filtering
                i+=1
                #days_done+=1 # Not working

            else:
                df_aux = pd.read_csv(directory+"/"+filename, sep='\t', header=None, names=column_names)

                df_aux = df_aux[df_aux['square_id'].isin(square_id)] # Id filtering
                df_aux = df_aux[df_aux['time_interval'].isin(time_interval)] # Time filtering

                df = df.append(df_aux)

                #days_done+=1 # Not working
    return df