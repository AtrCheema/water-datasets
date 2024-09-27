# -*- coding: utf-8 -*-
# Don't know which rights should be reserved  Ather Abbas
from setuptools import setup


import os
fpath = os.path.join(os.getcwd(), "readme.md")
if os.path.exists(fpath):
    with open(fpath, "r") as fd:
        long_desc = fd.read()
else:
    long_desc = "https://github.com/AtrCheema/water-datasets"


pandas_ver = 'pandas>=0.25.0, <= 2.1.4'


min_requirements = [
    pandas_ver,
    'requests',
    ]

extra_requires = [

"xarray",
"netCDF4",

# spatial processing
'imageio',
# shapely manually download the wheel file and install
'pyshp',

# for reading data
'netCDF4',
 'xarray',
]


all_requirements = min_requirements + extra_requires

setup(

    name='water_datasets',

    version="0.0.1",

    description='Platform for developing data driven based models for sequential/tabular data',
    long_description=long_desc,
    long_description_content_type="text/markdown",

    url='https://github.com/AtrCheema/water-datasets',

    author='Ather Abbas',
    author_email='ather_abbas786@yahoo.com',

    classifiers=[
        'Development Status :: 4 - Beta',

        'Natural Language :: English',

        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',

        'Operating System :: Microsoft :: Windows',
        'Operating System :: POSIX :: Linux',

        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
    ],

    packages=['water_datasets',
              'water_datasets/water_quality',
              'water_datasets/rr',
              ],

    install_requires=min_requirements,

    extras_require={
        'all': extra_requires,
    }
)
