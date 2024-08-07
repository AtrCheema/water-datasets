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