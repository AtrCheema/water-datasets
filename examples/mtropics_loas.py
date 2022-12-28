"""
================================
Rainfall-runoff dataset of Laos
================================
"""

# sphinx_gallery_thumbnail_number = -1

from easy_mpl import pie
from ai4water.eda import EDA
from ai4water.datasets import MtropicsLaos, ecoli_mekong

laos = MtropicsLaos(save_as_nc=False)

# %%
# precipitation
# --------------
pcp = laos.fetch_pcp()
print(pcp.shape)

# %%
# weather station
# -----------------
w = laos.fetch_weather_station_data()

# %%
wl, spm = laos.fetch_hydro()

# %%
ecoli = laos.fetch_ecoli()
print(ecoli.shape)

# %%
print(ecoli.head())

# %%
print(ecoli.tail())

# %%
ecoli_all = laos.fetch_ecoli(features='all')
print(ecoli_all.shape)

# %%
ecoli_all.head()

# %%
phy_chem = laos.fetch_physiochem('T_deg')
print(phy_chem.shape)

# %%
# pysiochemical attributes
# ------------------------
phy_chem_all = laos.fetch_physiochem(features='all')
print(phy_chem_all.shape)

# %%
# rain gauages
# -------------
rg = laos.fetch_rain_gauges()
print(rg.shape)

# %%
# regression
# -----------
df = laos.make_regression()
print(df.shape)

# %%
df.head()

# %%
df = laos.make_regression(lookback_steps=30)
print(df.shape)

# %%
df.head()

# %%
print(df.isna().sum())

# %%
eda = EDA(data=df)
eda.plot_data(subplots=True, figsize=(14, 20),
              ignore_datetime_index=True)

# %%
# classification
# -----------------------
df = laos.make_classification(lookback_steps=30)
print(df.shape)

# %%
df.head()

# %%
print(df.isna().sum())

# %%
# ecoli_mekong
# -------------
ecoli = ecoli_mekong()
print(ecoli.shape)

# %%
print(ecoli.head())

# %%
pie(df.dropna().iloc[:, -1].values.astype(int), explode=(0, 0.05))