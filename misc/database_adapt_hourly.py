# %%
"""
This code is used to take te "one region per day" data compilling to "one region per day, hourly". 
"""
import os
import sys
import glob

import numpy as np
import pandas as pd
from tqdm import tqdm

from datetime import datetime

sys.path.append('/home/patrick/Documents/milan-telecom-analysis-2022-10-11/libs')
import get_milano


# %%
# Paths to read the data and write adapted dataset
in_path = '/home/patrick/Documents/milan-telecom-analysis-2022-10-11/Dataset/telecom-sms,call,internet - per_cell'
out_path = '/home/patrick/Documents/milan-telecom-analysis-2022-10-11/Dataset/telecom-sms,call,internet - per_cell HOURLY'


# %%
# Verifique se o diretório de saída existe, se não, crie-o
if not os.path.exists(out_path):
    os.makedirs(out_path)

# Obtenha todos os arquivos CSV no diretório de entrada
csv_files = glob.glob(os.path.join(in_path, '*.csv'))

# Loop através de cada arquivo CSV
for csv_file in tqdm(csv_files):
    # Leia o CSV em um DataFrame
    df = pd.read_csv(csv_file, index_col=0)

    # Converta a coluna de timestamp para o tipo de data e hora
    df['timestamp'] = df.index
    df['timestamp'] = df['timestamp'].apply(lambda x: datetime.fromtimestamp(x/1000))
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Arredonde os timestamps para a hora mais próxima
    df['timestamp'] = df['timestamp'].dt.round('H')

    # Agrupe os registros por hora e some as colunas numéricas (se houver)
    df = df.groupby(['timestamp']).sum().reset_index()

    # Crie o nome do arquivo de saída
    base_name = os.path.basename(csv_file)
    out_file = os.path.join(out_path, base_name)

    # Salve o DataFrame resultante em um novo arquivo CSV
    df.to_csv(out_file, index=False)


# %%
