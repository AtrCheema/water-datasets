
import os
from typing import List, Union
import concurrent.futures as cf

import pandas as pd

from .camels import Camels
from .._backend import netCDF4
from .._backend import xarray as xr
from ..utils import check_attributes, get_cpus


BUL_COLUMNS = [
'snow_depth_water_equivalent_mean_BULL', 'surface_net_solar_radiation_mean_BULL', 'surface_net_thermal_radiation_mean_BULL', 
'surface_pressure_mean_BULL', 'temperature_2m_mean_BULL', 'dewpoint_temperature_2m_mean_BULL', 'u_component_of_wind_10m_mean_BULL',
 'v_component_of_wind_10m_mean_BULL', 'volumetric_soil_water_layer_1_mean_BULL', 'volumetric_soil_water_layer_2_mean_BULL', 
 'volumetric_soil_water_layer_3_mean_BULL', 'volumetric_soil_water_layer_4_mean_BULL', 'snow_depth_water_equivalent_min_BULL', 
 'surface_net_solar_radiation_min_BULL', 'surface_net_thermal_radiation_min_BULL', 'surface_pressure_min_BULL', 
 'temperature_2m_min_BULL', 'dewpoint_temperature_2m_min_BULL', 'u_component_of_wind_10m_min_BULL', 'v_component_of_wind_10m_min_BULL', 
 'volumetric_soil_water_layer_1_min_BULL', 'volumetric_soil_water_layer_2_min_BULL', 'volumetric_soil_water_layer_3_min_BULL', 
 'volumetric_soil_water_layer_4_min_BULL', 'snow_depth_water_equivalent_max_BULL', 'surface_net_solar_radiation_max_BULL', 
 'surface_net_thermal_radiation_max_BULL', 'surface_pressure_max_BULL', 'temperature_2m_max_BULL', 'dewpoint_temperature_2m_max_BULL', 
 'u_component_of_wind_10m_max_BULL', 'v_component_of_wind_10m_max_BULL', 'volumetric_soil_water_layer_1_max_BULL', 
 'volumetric_soil_water_layer_2_max_BULL', 'volumetric_soil_water_layer_3_max_BULL', 'volumetric_soil_water_layer_4_max_BULL', 
 'total_precipitation_sum_BULL', 'potential_evaporation_sum_BULL', 'streamflow_BULL'
]

class Bull(Camels):
    """
    Following the works of `Aparicio et al., 2024 <https://doi.org/10.1038/s41597-024-03594-5>`_.
    The data is taken from the `Zenodo repository <https://zenodo.org/records/10629809>`_.
    This dataset contains 484 stations with 55 dynamic (time series) features and
    214 static features. The dynamic features span from 1951 to 2021.

    Examples
    ---------
    >>> from water_datasets import Bull
    >>> dataset = Bull()
    >>> data = dataset.fetch(0.1, as_dataframe=True)
    >>> data.shape
    (1426260, 48)  # 40 represents number of stations
    Since data is a multi-index dataframe, we can get data of one station as below
    >>> data['BULL_9007'].unstack().shape  # the name of station could be different
    (25932, 13)
    If we don't set as_dataframe=True, then the returned data will be a xarray Dataset
    >>> data = dataset.fetch(0.1)
    >>> type(data)
        xarray.core.dataset.Dataset
    >>> data.dims
    FrozenMappingWarningOnValuesAccess({'time': 25932, 'dynamic_features': 55})
    >>> len(data.data_vars)
        48
    >>> df = dataset.fetch(stations=1, as_dataframe=True)  # get data of only one random station
    >>> df = df.unstack() # the returned dataframe is a multi-indexed dataframe so we have to unstack it
    >>> df.shape
    (25932, 55)
    # get name of all stations as list
    >>> stns = dataset.stations()
    >>> len(stns)
    484
    # get data by station id
    >>> df = dataset.fetch(stations='BULL_9007', as_dataframe=True).unstack()
    >>> df.shape
    (25932, 55)
    # get names of available dynamic features
    >>> dataset.dynamic_features
    # get only selected dynamic features
    >>> df = dataset.fetch(1, as_dataframe=True,
    ... dynamic_features=['potential_evapotranspiration_AEMET',  'temperature_mean_AEMET', 
    ... 'total_precipitation_ERA5_Land', 'streamflow']).unstack()
    >>> df.shape
    (25932, 4)
    # get names of available static features
    >>> dataset.static_features
    # get data of 10 random stations
    >>> df = dataset.fetch(10, as_dataframe=True)
    >>> df.shape
    (166166, 10)  # remember this is multi-indexed DataFrame
    # when we get both static and dynamic data, the returned data is a dictionary
    # with ``static`` and ``dyanic`` keys.
    >>> data = dataset.fetch(stations='BULL_9007', static_features="all", as_dataframe=True)
    >>> data['static'].shape, data['dynamic'].shape
    ((1, 214), (1426260, 1))
    >>> coords = dataset.stn_coords() # returns coordinates of all stations
    >>> coords.shape
        (484, 2)
    >>> dataset.stn_coords('BULL_9007')  # returns coordinates of station whose id is GRDC_3664802
        41.298	-1.967
    >>> dataset.stn_coords(['BULL_9007', 'BULL_8083'])  # returns coordinates of two stations      
    
    """

    url = "https://zenodo.org/records/10629809"

    def __init__(
            self, 
            path, 
            overwrite=False,
            **kwargs
            ):
        super().__init__(path, **kwargs)

        self._download(overwrite=overwrite)

        if netCDF4 is None:
            self.ftype = "csv"
        else:
            self.ftype = "netcdf"
        
        self._dynamic_features = self._read_dynamic_for_stn(self.stations()[0]).columns.tolist()
        self._static_features = list(set(self.static_data().columns.tolist()))

        self.boundary_file = os.path.join(self.shapefiles_path, "BULL_basin_shapes.shp")

        self._create_boundary_id_map(self.boundary_file, 0)

        self.dyn_fname = ''
    
    @property
    def attributes_path(self):
        return os.path.join(self.path, "attributes", "attributes")

    @property
    def shapefiles_path(self):
        return os.path.join(self.path, "shapefiles", "shapefiles")
    
    @property
    def ts_path(self):
        return os.path.join(self.path, "timeseries", "timeseries")
    
    @property
    def q_path(self):
        return os.path.join(self.ts_path, self.ftype, "streamflow")
    
    @property
    def aemet_path(self):
        return os.path.join(self.ts_path, self.ftype, "AEMET")

    @property
    def bull_path(self):
        return os.path.join(self.ts_path, self.ftype, "BULL")
    
    @property
    def era5_land_path(self):
        return os.path.join(self.ts_path, self.ftype, "ERA5_Land")
    
    @property
    def emo1_arc_path(self):
        return os.path.join(self.ts_path, self.ftype, "EMO1_arc")

    @property
    def _q_name(self)->str:
        return "streamflow"
    
    @property
    def _coords_name(self)->List[str]:
        return ['gauge_lat', 'gauge_lon']

    @property
    def _area_name(self) ->str:
        return 'area' 
    
    @property
    def start(self):
        return pd.Timestamp("19510102")

    @property
    def end(self):
        return pd.Timestamp("20211231")
    
    def stations(self)->List[str]:
        return ["BULL_" + f.split('.')[0].split('_')[1] for f in os.listdir(self.q_path)]
    
    @property
    def dynamic_features(self)->List[str]:
        return self._dynamic_features
    
    @property
    def static_features(self)->List[str]:
        return self._static_features

    def caravan_attributes(self)->pd.DataFrame:
        """a dataframe of shape (484, 10)"""
        return pd.read_csv(
            os.path.join(self.attributes_path, "attributes_caravan_.csv"), 
            index_col=0)
    
    def hydroatlas_attributes(self)->pd.DataFrame:
        """a dataframe of shape (484, 197)"""
        df = pd.read_csv(
            os.path.join(self.attributes_path, "attributes_hydroatlas_.csv"), 
            index_col=0)
        # because self.other_attributes() has a column named 'area'
        df.rename(columns={'area': 'area_hydroatlas'}, inplace=True)
        return df
    
    def other_attributes(self)->pd.DataFrame:
        """a dataframe of shape (484, 7)"""
        return pd.read_csv(
            os.path.join(self.attributes_path, "attributes_other_ss.csv"), 
            index_col=0)
    
    def static_data(self)->pd.DataFrame:
        return pd.concat([
            self.caravan_attributes(),
            self.hydroatlas_attributes(),
            self.other_attributes()
        ], axis=1)

    def _read_dynamic_for_stn(self, stn_id:str)->pd.DataFrame:

        stn_id = stn_id.split('_')[1]

        df = pd.concat([
            self._read_q_for_stn(stn_id),
            self._read_aemet_for_stn(stn_id),
            self._read_bull_for_stn(stn_id),
            self._read_era5_land_for_stn(stn_id),
            self._read_emo1_arc_for_stn(stn_id)
        ], axis=1)
        df.index.name = 'time'
        df.columns.name = 'dynamic_features'        
        return df    

    def _read_dynamic_from_csv(
            self, 
            stations, 
            dynamic_features, 
            st=None,
            en=None)->dict:

        dynamic_features = check_attributes(dynamic_features, self.dynamic_features)
        stations = check_attributes(stations, self.stations())

        if st is None:
            st = self.start
        if en is None:
            en = self.end

        cpus = self.processes or min(get_cpus(), 64)

        if len(stations) > 10 and cpus > 1:

            with  cf.ProcessPoolExecutor(max_workers=cpus) as executor:
                results = executor.map(
                    self._read_dynamic_for_stn,
                    stations,
                )
            dyn = {stn:data.loc[st:en, dynamic_features] for stn, data in zip(stations, results)}
        else:
            dyn = {
                stn: self._read_dynamic_for_stn(stn).loc[st: en, dynamic_features] for stn in stations
            }

        return dyn

    def _read_q_for_stn(self, stn_id)->pd.DataFrame:
        """a dataframe of shape (time, 1)"""
        if self.ftype == "netcdf":
            fpath = os.path.join(self.q_path, f'streamflow_{stn_id}.nc')
            df = xr.load_dataset(fpath).to_dataframe()
        else:
            fpath = os.path.join(self.q_path, f'streamflow_{stn_id}.csv')
            df = pd.read_csv(fpath, index_col='date', parse_dates=True)
        df.index.name = 'time'
        df.columns.name = 'dynamic_features'
        return df       
    
    def _read_aemet_for_stn(self, stn_id)->pd.DataFrame:
        """a dataframe of shape (time, 5)
        'temperature_max_AEMET', 
        'temperature_min_AEMET', 
        'temperature_mean_AEMET', 
        'total_precipitation_AEMET', 
        'potential_evapotranspiration_AEMET'
        """
        if self.ftype == "netcdf":
            fpath = os.path.join(self.aemet_path, f'AEMET_{stn_id}.nc')
            df = xr.load_dataset(fpath).to_dataframe()
        else:
            fpath = os.path.join(self.aemet_path, f'AEMET_{stn_id}.csv')
            df = pd.read_csv(fpath, index_col='date', parse_dates=True)
        df.index.name = 'time'
        df.columns.name = 'dynamic_features'
        df.columns = [col+'_AEMET' for col in df.columns]
        return df
    
    def _read_bull_for_stn(self, stn_id)->pd.DataFrame:
        """a dataframe of shape (time, 39) except for stn 3163"""
        if self.ftype == "netcdf":
            fpath = os.path.join(self.bull_path, f'BULL_{stn_id}.nc')
            df = xr.load_dataset(fpath).to_dataframe()
        else:
            fpath = os.path.join(self.bull_path, f'BULL_{stn_id}.csv')
            df = pd.read_csv(fpath, index_col='date', parse_dates=True)
        df.index.name = 'time'
        df.columns.name = 'dynamic_features'
        df.columns = [col+'_BULL' for col in df.columns]
        if len(df.columns) == 15:
            # add missing columns
            for col in BUL_COLUMNS:
                if col not in df.columns:
                    df[col] = None
        return df
    
    def _read_era5_land_for_stn(self, stn_id)->pd.DataFrame:
        """a dataframe of shape (time, 5) with following columns
            - 'temperature_max_ERA5_Land', 
            - 'temperature_min_ERA5_Land', 
            - 'temperature_mean_ERA5_Land', 
            - 'total_precipitation_ERA5_Land', 
            - 'potential_evapotranspiration_ERA5_Land'
        """
        if self.ftype == "netcdf":
            fpath = os.path.join(self.era5_land_path, f'ERA5_Land_{stn_id}.nc')
            df = xr.load_dataset(fpath).to_dataframe()
        else:
            fpath = os.path.join(self.era5_land_path, f'ERA5_Land_{stn_id}.csv')
            df = pd.read_csv(fpath, index_col='date', parse_dates=True)
        df.index.name = 'time'
        df.columns.name = 'dynamic_features'
        df.columns = [col+'_ERA5_Land' for col in df.columns]
        return df
    
    def _read_emo1_arc_for_stn(self, stn_id)->pd.DataFrame:
        """a dataframe of shape (time, 5) with following columns
            - 'temperature_max_EMO1_arc' 
            - 'temperature_min_EMO1_arc' 
            - 'temperature_mean_EMO1_arc'
            - 'total_precipitation_EMO1_arc'
            - 'potential_evapotranspiration_EMO1_arc'
        """
        if self.ftype == "netcdf":
            fpath = os.path.join(self.emo1_arc_path, f'EMO1_{stn_id}.nc')
            df = xr.load_dataset(fpath).to_dataframe()
        else:
            fpath = os.path.join(self.emo1_arc_path, f'EMO1_{stn_id}.csv')
            df = pd.read_csv(fpath, index_col='date', parse_dates=True)
        df.index.name = 'time'
        df.columns.name = 'dynamic_features'
        df.columns = [col+'_EMO1_arc' for col in df.columns]
        return df

    def fetch_static_features(
            self,
            stn_id: Union[str, List[str]] = None,
            features:Union[str, List[str]]=None
    ) -> pd.DataFrame:
        """
        Returns static features of one or more stations.

        Parameters
        ----------
            stn_id : str
                name/id of station/stations of which to extract the data
            features : list/str, optional (default="all")
                The name/names of features to fetch. By default, all available
                static features are returned.

        Returns
        -------
        pd.DataFrame
            a pandas dataframe of shape (stations, features)

        Examples
        ---------
        >>> from water_datasets import Bull
        >>> dataset = Bull()
        get the names of stations
        >>> stns = dataset.stations()
        >>> len(stns)
            484
        get all static data of all stations
        >>> static_data = dataset.fetch_static_features(stns)
        >>> static_data.shape
           (484, 214)
        get static data of one station only
        >>> static_data = dataset.fetch_static_features('42600042')
        >>> static_data.shape
           (1, 214)
        get the names of static features
        >>> dataset.static_features
        get only selected features of all stations
        >>> static_data = dataset.fetch_static_features(stns, ['seasonality', 'moisture_index'])
        >>> static_data.shape
           (484, 2)
        >>> data = dataset.fetch_static_features('42600042', features=['seasonality', 'moisture_index'])
        >>> data.shape
           (1, 2)

        """
        stations = check_attributes(stn_id, self.stations())
        features = check_attributes(features, self.static_features)
        df = self.static_data()
        return df.loc[stations, features]