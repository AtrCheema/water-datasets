"""
====================
CAMELS Australia
====================
"""
import os
import site

if __name__ == '__main__':
    wd_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath('__file__')))))
    # wd_dir = os.path.dirname(os.path.dirname(os.path.realpath('__file__')))
    print(wd_dir)
    site.addsitedir(wd_dir)

from water_datasets import CAMELS_AUS
from water_datasets.utils import print_info

# %%

print_info()

# %%

dataset = CAMELS_AUS()

# %%

df = dataset.fetch(stations=1, as_dataframe=True)
df = df.unstack() # the returned dataframe is a multi-indexed dataframe so we have to unstack it
df.shape

# %%

# get name of all stations as list
stns = dataset.stations()
len(stns)

# %%
# get data of 10 % of stations as dataframe
df = dataset.fetch(0.1, as_dataframe=True)
df.shape

# %%

df

# %%

# The returned dataframe is a multi-indexed data
df.index.names == ['time', 'dynamic_features']

df
# %%

# get data by station id
df = dataset.fetch(stations='224214A', as_dataframe=True).unstack()
df.shape

# %%

df

# %%

# get names of available dynamic features
dataset.dynamic_features
# get only selected dynamic features
data = dataset.fetch(1, as_dataframe=True,
dynamic_features=['tmax_AWAP', 'precipitation_AWAP', 'et_morton_actual_SILO', 'streamflow_MLd']).unstack()
data.shape

# %%

data

# %%

# get names of available static features
dataset.static_features
# get data of 10 random stations
df = dataset.fetch(10, as_dataframe=True)
df.shape  # remember this is a multiindexed dataframe

# %%

# when we get both static and dynamic data, the returned data is a dictionary
# with ``static`` and ``dyanic`` keys.
data = dataset.fetch(stations='224214A', static_features="all", as_dataframe=True)
data['static'].shape, data['dynamic'].shape

# %%
data['static']

# %%

data['dynamic']
