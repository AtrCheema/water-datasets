"""
=======
GRQA
=======
"""


# from ai4water.datasets import GRQA
# from easy_mpl import plot
# import matplotlib.pyplot as plt
#
# # %%
# import ai4water
# print(ai4water.__version__)
#
# # %%
# ds = GRQA()
#
# # %%
# print(ds.parameters)
#
# # %%
# country = "Pakistan"
# len(ds.fetch_parameter('TEMP', country=country))
#
# # %%
# params = []
# for param in ds.parameters:
#     if len(ds.fetch_parameter(param, country=country))>1:
#         params.append(param)
#
# print(params)

# %%

# df = ds.fetch_parameter("TEMP", country=country)
# print(df.shape)
# for grp_name, grp in df.groupby('site_name'):
#     plot(grp['obs_value'], '--.', label=grp_name, show=False,
#          ax_kws=dict(figsize=(18,7)))
# plt.show()
#
# # %%
# df = ds.fetch_parameter("NH4N", country=country)
# print(df.shape)
# for grp_name, grp in df.groupby('site_name'):
#     plot(grp['obs_value'], '--.', label=grp_name, show=False,
#          ax_kws=dict(figsize=(18,7)))
# plt.show()
#
# # %%
# df = ds.fetch_parameter("DO", country=country)
# print(df.shape)
# for grp_name, grp in df.groupby('site_name'):
#     plot(grp['obs_value'], '--.', label=grp_name, show=False,
#          ax_kws=dict(figsize=(18,7)))
# plt.show()
#
# # %%
# df = ds.fetch_parameter("COD", country=country)
# print(df.shape)
# for grp_name, grp in df.groupby('site_name'):
#     plot(grp['obs_value'], '--.', label=grp_name, show=False,
#          ax_kws=dict(figsize=(18,7)))
# plt.show()
#
# # %%
# df = ds.fetch_parameter("BOD", country=country)
# print(df.shape)
# for grp_name, grp in df.groupby('site_name'):
#     plot(grp['obs_value'], '--.', label=grp_name, show=False,
#          ax_kws=dict(figsize=(18,7)))
# plt.show()
#
# # %%
# df = ds.fetch_parameter("DON", country=country)
# print(df.shape)
# for grp_name, grp in df.groupby('site_name'):
#     plot(grp['obs_value'], '--.', label=grp_name, show=False,
#          ax_kws=dict(figsize=(18,7)))
# plt.show()
#
# # %%
# df = ds.fetch_parameter("DOSAT", country=country)
# print(df.shape)
# for grp_name, grp in df.groupby('site_name'):
#     plot(grp['obs_value'], '--.', label=grp_name, show=False,
#          ax_kws=dict(figsize=(18,7)))
# plt.show()
#
# # %%
# df = ds.fetch_parameter("TDP", country=country)
# print(df.shape)
# for grp_name, grp in df.groupby('site_name'):
#     plot(grp['obs_value'], '--.', label=grp_name, show=False,
#          ax_kws=dict(figsize=(18,7)))
# plt.show()
#
# # %%
# df = ds.fetch_parameter("TKN", country=country)
# print(df.shape)
# for grp_name, grp in df.groupby('site_name'):
#     plot(grp['obs_value'], '--.', label=grp_name, show=False,
#          ax_kws=dict(figsize=(18,7)))
# plt.show()
#
# # %%
# df = ds.fetch_parameter("TSS", country=country)
# print(df.shape)
# for grp_name, grp in df.groupby('site_name'):
#     plot(grp['obs_value'], '--.', label=grp_name, show=False,
#          ax_kws=dict(figsize=(18,7)))
# plt.show()
#
# # %%
# df = ds.fetch_parameter("TP", country=country)
# print(df.shape)
# for grp_name, grp in df.groupby('site_name'):
#     plot(grp['obs_value'], '--.', label=grp_name, show=False,
#          ax_kws=dict(figsize=(18,7)))
# plt.show()
#
# # %%
# df = ds.fetch_parameter("pH", country=country)
# print(df.shape)
# for grp_name, grp in df.groupby('site_name'):
#     plot(grp['obs_value'], '--.', label=grp_name, show=False,
#          ax_kws=dict(figsize=(18,7)))
# plt.show()

# %%
