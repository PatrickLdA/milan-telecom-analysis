# %%
import os 
import pandas as pd 
import matplotlib.pyplot as plt
%matplotlib inline
import numpy as np
import pickle

# %%
path = r'C:\\Users\\patri\\Documents\\Github\\milan-telecom-analysis\\results\\compiled_results'
path_time = r'C:\\Users\\patri\\Documents\\Github\\milan-telecom-analysis\\results\\compile_time'

# %%
# GENERATING ONE VIEW FOR EACH  MODEL
for filename in os.listdir(path):
    if filename[-5:] != 'E.csv':
        cols = list(pd.read_csv(os.path.join(path, filename), nrows=1))
        neighbors = pd.read_csv(os.path.join(path, filename), usecols =[i for i in cols if i != 'Unnamed: 0'])

        plt.figure()
        neighbors.plot(figsize=[15,10], legend=None)

        plt.xlabel('Epoch')
        plt.ylabel('NMAE')
        plt.yscale('log')
        plt.title(f'Evolution for {filename[0]} neighbors', fontsize=20)

        plt.savefig(os.path.join(path,f'{filename[:-4]}.png'))
        #plt.show()

# %%
first = True
for filename in os.listdir(path):
    if filename[-5:] == 's.csv':
        if first:
            first = False

            cols = list(pd.read_csv(os.path.join(path, filename), nrows=1))
            compiled_results = pd.DataFrame(pd.read_csv(os.path.join(path, filename), usecols =[i for i in cols if i != 'Unnamed: 0']).iloc[-1])

            compiled_results.columns = [filename[:-4]]

        else:
            cols = list(pd.read_csv(os.path.join(path, filename), nrows=1))
            aux = pd.read_csv(os.path.join(path, filename), usecols =[i for i in cols if i != 'Unnamed: 0']).iloc[-1]

            compiled_results[filename[:-4]] = aux

plt.figure(figsize=[10,7])

compiled_results.plot(figsize=[10,7]
                        , kind='box'
                        , color=dict(boxes='r', whiskers='r', medians='r', caps='r')
                        , boxprops=dict(linestyle='-', linewidth=1.8)
                        , flierprops=dict(linestyle='-', linewidth=1.8)
                        , medianprops=dict(linestyle='-', linewidth=1.8)
                        , whiskerprops=dict(linestyle='-', linewidth=1.8)
                        , capprops=dict(linestyle='-', linewidth=1.8)
                        , showfliers=False
                        , grid=True
                        , rot=0)

plt.ylabel('NMAE na época 50', fontsize=15)
plt.yticks(size=10)
plt.xlabel('Vizinhanças', fontsize=15)
plt.xticks(size=10)
plt.title(f'NMAE para n vizinhanças', fontsize=20)

plt.savefig(os.path.join(path,'Boxplot NMAE.png'))
#plt.show()

# %%
first = True
for filename in os.listdir(path):
    if filename[-5:] == 'c.csv':
        if first:
            first = False

            cols = list(pd.read_csv(os.path.join(path, filename), nrows=1))
            compiled_results = pd.DataFrame(pd.read_csv(os.path.join(path, filename), usecols =[i for i in cols if i != 'Unnamed: 0']).iloc[-1])

            compiled_results.columns = [filename[:11]]

        else:
            cols = list(pd.read_csv(os.path.join(path, filename), nrows=1))
            aux = pd.read_csv(os.path.join(path, filename), usecols =[i for i in cols if i != 'Unnamed: 0']).iloc[-1]

            compiled_results[filename[:11]] = aux

plt.figure(figsize=[10,7])
compiled_results.plot(figsize=[10,7]
                        , kind='box'
                        , color=dict(boxes='r', whiskers='r', medians='r', caps='r')
                        , boxprops=dict(linestyle='-', linewidth=1.8)
                        , flierprops=dict(linestyle='-', linewidth=1.8)
                        , medianprops=dict(linestyle='-', linewidth=1.8)
                        , whiskerprops=dict(linestyle='-', linewidth=1.8)
                        , capprops=dict(linestyle='-', linewidth=1.8)
                        , showfliers=False
                        , grid=True
                        , rot=0)
plt.legend(loc='best')
plt.ylabel('NMAE na época 50', fontsize=15)
plt.yticks(size=10)
plt.xlabel('Vizinhanças', fontsize=15)
plt.xticks(size=10)
plt.title(f'NMAE para n vizinhanças (considerando hubs de transporte)', fontsize=20)

plt.savefig(os.path.join(path,'Boxplot NMAE feature selec.png'))
#plt.show()

# %%
df_time = pd.DataFrame(columns=['With_transport_hubs', 'No_transport_hubs'], index=['1', '2', '3', '4', '5'])

for filename in os.listdir(path_time):
    if filename[-12:] == 'selec.pickle':
        with open(os.path.join(path_time, filename), 'rb') as filepath:
            var = pickle.load(filepath)

        df_time.loc[filename[0], 'With_transport_hubs'] = np.mean(var)
        

    if filename[-11:] == 'time.pickle':
        with open(os.path.join(path_time, filename), 'rb') as filepath:
            var = pickle.load(filepath)

        df_time.loc[filename[0], 'No_transport_hubs'] = np.mean(var)

        
# %%
plt.figure(figsize=[23,10])

df_time.plot(figsize=[23,10], marker='o', linewidth=4, ms=15)

plt.legend(loc='best', prop={'size': 25})
plt.ylabel('Tempo médio (segundos)', fontsize=25)
plt.yticks(size=15)
plt.xlabel('Vizinhanças', fontsize=25)
plt.xticks(size=15)
plt.title(f'Tempo médio de execução por número de vizinhanças usado', fontsize=20)

plt.savefig(os.path.join(path_time,'Execution time.png'))

# %%
