"""
===========================
beach water quality
===========================
"""

import numpy as np

np.bool = np.bool_

import os
import site

if __name__ == '__main__':
    wd_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath('__file__')))))
    print(wd_dir)
    site.addsitedir(wd_dir)

from ai4water.eda import EDA
from water_datasets import busan_beach
from water_datasets.utils import print_info

# sphinx_gallery_thumbnail_number = 7

print_info()
###########################################################


data = busan_beach(target=['ecoli', 'sul1_coppml', 'aac_coppml',
                           'tetx_coppml', 'blaTEM_coppml'])
print(data.shape)

###########################################################

data.head()

###########################################################

data.isna().sum()

###########################################################

data.isna().sum()

###########################################################

eda = EDA(data, save=False)

###########################################################

eda.heatmap()

###########################################################

_ = eda.plot_missing()

###########################################################

# _ = eda.plot_data(subplots=True, max_cols_in_plot=20, figsize=(14, 20))
#
# ###########################################################

eda.plot_data(subplots=True, max_cols_in_plot=20, figsize=(14, 20),
              ignore_datetime_index=True)

###########################################################

_ = eda.plot_histograms()

###########################################################

eda.box_plot(max_features=18, palette="Set3")

###########################################################

eda.box_plot(max_features=18, palette="Set3", violen=True)

###########################################################

eda.correlation(figsize=(14, 14))

# ###########################################################
#
#
# eda.grouped_scatter(max_subplots=18)

###########################################################

_ = eda.autocorrelation(n_lags=15)

###########################################################

_ = eda.partial_autocorrelation(n_lags=15)

###########################################################

_ = eda.lag_plot(n_lags=14, s=0.4)

############################################################


_ = eda.plot_ecdf(figsize=(10, 14))

############################################################

eda.normality_test()
