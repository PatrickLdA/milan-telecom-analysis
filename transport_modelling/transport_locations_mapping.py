# %%
import json
import pandas as pd
from shapely.geometry import shape, Point
import numpy as np

# %%
transport_path = 'public_transport_locations.csv'
grid_path = r'C:\\Users\\patri\\Documents\\Dataset\\milano-grid.geojson'

# %%
# Grid load
with open(grid_path) as f:
    grid = json.load(f)

# %%
# Location map load
locs = pd.read_csv(transport_path, index_col=0)

# %%
# Point-based construction
res = []

for index, rows in locs.iterrows():
    point = Point(rows['lng'], rows['lat'])

    for feature in grid['features']:
        polygon = shape(feature['geometry'])

        if polygon.contains(point):
            res.append([rows['loc'], feature['properties']['cellId']])

# %%
locs['cell_id'] = np.NaN

for reg in res:
    locs.loc[(locs['loc'] == reg[0]), 'cell_id']  = int(reg[1])

# %%
locs = locs.dropna().reset_index(drop=True)

# %%
locs.to_csv(transport_path)

# %%
