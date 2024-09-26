"""
Rainfall Runoff datasets
"""

import os
from typing import Union, List

import pandas as pd
from .._backend import xarray as xr, plt, plt_Axes

from .camels import Camels
from ._camels import CAMELS_AUS
from ._camels import CAMELS_CL
from ._camels import CAMELS_GB
from ._camels import CAMELS_US
from ._lamah import LamaH
from ._brazil import CAMELS_BR
from ._brazil import CABra
from ._hysets import HYSETS
from ._hype import HYPE
from ._camels import CAMELS_DK
from ._waterbenchiowa import WaterBenchIowa
from ._gsha import GSHA
from ._ccam import CCAM
from ._rrluleasweden import RRLuleaSweden
from ._camels import CAMELS_CH
from ._lamah import LamaHIce
from ._camels import CAMELS_DE
from ._camels import GRDCCaravan
from ._camels import CAMELS_SE
from ._simbi import Simbi
from ._denmark import CAMELS_DK as CAMELS_DK0
from ._bull import Bull
from ._camels import CAMELS_IND


DATASETS = {
    "camels": Camels,
    "CAMELS_AUS": CAMELS_AUS,
    "CAMELS_CL": CAMELS_CL,
    "CAMELS_GB": CAMELS_GB,
    "CAMELS_US": CAMELS_US,
    "LamaH": LamaH,
    "CAMELS_BR": CAMELS_BR,
    "CABra": CABra,
    "HYSETS": HYSETS,
    "HYPE": HYPE,
    "CAMELS_DK": CAMELS_DK,
    "WaterBenchIowa": WaterBenchIowa,
    "GSHA": GSHA,
    "CCAM": CCAM,
    "RRLuleaSweden": RRLuleaSweden,
    "CAMELS_CH": CAMELS_CH,
    "LamaHIce": LamaHIce,
    "CAMELS_DE": CAMELS_DE,
    "GRDCCaravan": GRDCCaravan,
    "CAMELS_SE": CAMELS_SE,
    "Simbi": Simbi,
    "CAMELS_DK0": CAMELS_DK0,
    "Bull": Bull,
    "CAMELS_IND": CAMELS_IND
}


class RainfallRunoff(object):
    """
    Rainfall Runoff datasets
    """
    def __init__(
            self,
            dataset:str,
            path: Union[str, os.PathLike],
            overwrite:bool = False,
            to_netcdf:bool = True,
            processes:int = None,
            remove_zip:bool = True,
            verbosity:int = 1,
            **kwargs
    ):
        """
        Rainfall Runoff datasets

        Parameters
        ----------
        dataset: str
            dataset name. This must be one of the following:
    
             - `CAMELS_AUS`
             - `CAMELS_CL`
             - `CAMELS_GB`
            - `CAMELS_US`
            - `LamaH`
            - `CAMELS_BR`
            - `CABra`
            - `HYSETS`
            - `HYPE`
            - `CAMELS_DK`
            - `WaterBenchIowa`
            - `GSHA`
            - `CCAM`
            - `RRLuleaSweden`
            - `CAMELS_CH`
            - `LamaHIce`
            - `CAMELS_DE`
            - `GRDCCaravan`
            - `CAMELS_SE`
            - `Simbi`
            - `CAMELS_DK0`
            - `Bull`
            - `CAMELS_IND`
        path : str
            path to directory where the data is stored or where the data is to be
            downloaded. If provided and the path exists, then the data will be read
            from this path. If provided and the path does not exist,
            then the data will be downloaded in this path. If not provided,
            then the data will be downloaded in the default path.
        overwrite : bool
            If the data is already downloaded then you can set it to True,
            to make a fresh download.
        to_netcdf : bool
            whether to convert all the data into one netcdf file or not.
            This will fasten repeated calls to fetch etc but will
            require netcdf5 package as well as xarray.
        verbosity : int
            0: no message will be printed
        kwargs :
            additional keyword arguments for the underlying dataset class
            For example ``version`` for CAMELS_AUS or ``timestep`` for 
            LamaH dataset.
        """

        if dataset not in DATASETS:
            raise ValueError(f"Dataset {dataset} not available")

        self.dataset = DATASETS[dataset](
            path=path, 
            overwrite=overwrite, 
            to_netcdf=to_netcdf,
            processes=processes,
            remove_zip=remove_zip,
            verbosity=verbosity,
            **kwargs
            )

    def __repr__(self):
        return f"RainfallRunoff({self.dataset}) with {len(self.stations())} stations, {self.num_dynamic} dynamic features and {self.num_static} static features"

    def __str__(self):
        return f"RainfallRunoff({self.dataset}) with {len(self.stations())} stations, {self.num_dynamic} dynamic features and {self.num_static} static features"

    def __len__(self):
        return len(self.stations())

    def num_dynamic(self)->int:
        return len(self.dynamic_features)

    def num_static(self)->int:
        return len(self.static_features)

    @property
    def name(self)->str:
        """
        returns name of dataset
        """
        return self.dataset.name
    
    @property
    def path(self)->str:
        """
        returns path of dataset
        """
        return self.dataset.path

    @property
    def static_features(self)->List[str]:
        """
        returns names of static features
        """
        return self.dataset.static_features
    
    @property
    def dynamic_features(self)->List[str]:
        """
        returns names of dynamic features
        """
        return self.dataset.dynamic_features

    def fetch_static_features(
            self,
            stn_id: Union[str, list] = None,
            features: Union[str, list] = None
    )->pd.DataFrame:
        """Fetches all or selected static attributes of one or more stations.

        Parameters
        ----------
            stn_id : str
                name/id of station of which to extract the data
            features : list/str, optional (default="all")
                The name/names of features to fetch. By default, all available
                static features are returned.

        Returns
        -------
        pd.DataFrame
            a pandas dataframe

        Examples
        --------
            >>> from water_datasets import RainfallRunoff
            >>> camels = RainfallRunoff('CAMELS_AUS')
            >>> camels.fetch_static_features('224214A')
            >>> camels.static_features
            >>> camels.fetch_static_features('224214A',
            ... features=['elev_mean', 'relief', 'ksat', 'pop_mean'])
        """

        return self.dataset.fetch_static_features(stn_id, features)

    def area(
            self,
            stations: Union[str, List[str]] = None
    ) ->pd.Series:
        """
        Returns area (Km2) of all/selected catchments as pandas series

        parameters
        ----------
        stations : str/list (default=None)
            name/names of stations. Default is None, which will return
            area of all stations

        Returns
        --------
        pd.Series
            a pandas series whose indices are catchment ids and values
            are areas of corresponding catchments.

        Examples
        ---------
        >>> from water_datasets import RainfallRunoff
        >>> dataset = RainfallRunoff('CAMELS_CH')
        >>> dataset.area()  # returns area of all stations
        >>> dataset.area('2004')  # returns area of station whose id is 2004
        >>> dataset.area(['2004', '6004'])  # returns area of two stations
        """
        return self.dataset.area(stations)


    def fetch(self,
              stations: Union[str, list, int, float, None] = None,
              dynamic_features: Union[list, str, None] = 'all',
              static_features: Union[str, list, None] = None,
              st: Union[None, str] = None,
              en: Union[None, str] = None,
              as_dataframe: bool = False,
              **kwargs
              ) -> Union[dict, pd.DataFrame]:
        """
        Fetches the attributes of one or more stations.

        parameters
        ----------
        stations : 
            if string, it is supposed to be a station name/gauge_id.
            If list, it will be a list of station/gauge_ids. If int, it will
            be supposed that the user want data for this number of
            stations/gauge_ids. If None (default), then attributes of all
            available stations. If float, it will be supposed that the user
            wants data of this fraction of stations.
        dynamic_features : 
            If not None, then it is the attributes to be
            fetched. If None, then all available attributes are fetched
        static_features : 
            list of static attributes to be fetches. None
            means no static attribute will be fetched.
        st : 
            starting date of data to be returned. If None, the data will be
            returned from where it is available.
        en : 
            end date of data to be returned. If None, then the data will be
            returned till the date data is available.
        as_dataframe : 
            whether to return dynamic attributes as pandas
            dataframe or as xarray dataset.
        kwargs : keyword arguments to read the files

        returns:
            If both static  and dynamic features are obtained then it returns a
            dictionary whose keys are station/gauge_ids and values are the
            attributes and dataframes.
            Otherwise either dynamic or static features are returned.

        Examples
        --------
        >>> from water_datasets import RainfallRunoff
        >>> dataset = RainfallRunoff('CAMELS_AUS')
        >>> # get data of 10% of stations
        >>> df = dataset.fetch(stations=0.1, as_dataframe=True)  # returns a multiindex dataframe
        ...  # fetch data of 5 (randomly selected) stations
        >>> five_random_stn_data = dataset.fetch(stations=5, as_dataframe=True)
        ... # fetch data of 3 selected stations
        >>> three_selec_stn_data = dataset.fetch(stations=['912101A','912105A','915011A'], as_dataframe=True)
        ... # fetch data of a single stations
        >>> single_stn_data = dataset.fetch(stations='318076', as_dataframe=True)
        ... # get both static and dynamic features as dictionary
        >>> data = dataset.fetch(1, static_features="all", as_dataframe=True)  # -> dict
        >>> data['dynamic']
        ... # get only selected dynamic features
        >>> sel_dyn_features = dataset.fetch(stations='318076',
        ...     dynamic_features=['streamflow_MLd', 'solarrad_AWAP'], as_dataframe=True)
        ... # fetch data between selected periods
        >>> data = dataset.fetch(stations='318076', st="20010101", en="20101231", as_dataframe=True)

        """
        return self.dataset.fetch(stations, dynamic_features, static_features, st, en, as_dataframe, **kwargs)


    def fetch_stations_features(
            self,
            stations: Union[str, List[str]],
            dynamic_features='all',
            static_features=None,
            st=None,
            en=None,
            as_dataframe: bool = False,
            **kwargs
    ):
        """Reads attributes of more than one stations.

        parameters
        ----------
        stations : 
            list of stations for which data is to be fetched.
        dynamic_features : 
            list of dynamic attributes to be fetched.
                if 'all', then all dynamic attributes will be fetched.
        static_features : 
            list of static attributes to be fetched.
                If `all`, then all static attributes will be fetched. If None,
                then no static attribute will be fetched.
        st : 
            start of data to be fetched.
        en : 
            end of data to be fetched.
        as_dataframe : whether to return the data as pandas dataframe. default
                is xr.Dataset object
        kwargs dict: additional keyword arguments

        Returns:
            Dynamic and static features of multiple stations. Dynamic features
            are by default returned as xr.Dataset unless `as_dataframe` is True, in
            such a case, it is a pandas dataframe with multiindex. If xr.Dataset,
            it consists of `data_vars` equal to number of stations and for each
            station, the `DataArray` is of dimensions (time, dynamic_features).
            where `time` is defined by `st` and `en` i.e. length of `DataArray`.
            In case, when the returned object is pandas DataFrame, the first index
            is `time` and second index is `dyanamic_features`. Static attributes
            are always returned as pandas DataFrame and have following shape
            `(stations, static_features). If `dynamic_features` is None,
            then they are not returned and the returned value only consists of
            static features. Same holds true for `static_features`.
            If both are not None, then the returned type is a dictionary with
            `static` and `dynamic` keys.

        Raises:
            ValueError, if both dynamic_features and static_features are None

        Examples
        --------
            >>> from water_datasets import RainfallRunoff
            >>> dataset = RainfallRunoff('CAMELS_AUS')
            ... # find out station ids
            >>> dataset.stations()
            ... # get data of selected stations
            >>> dataset.fetch_stations_features(['912101A', '912105A', '915011A'],
            ...  as_dataframe=True)
        """
        return self.dataset.fetch_stations_features(stations, dynamic_features, static_features, st, en, as_dataframe, **kwargs)

    def fetch_dynamic_features(
            self,
            stn_id: str,
            features='all',
            st=None,
            en=None,
            as_dataframe=False
    ):
        """Fetches all or selected dynamic attributes of one station.

        Parameters
        ----------
            stn_id : str
                name/id of station of which to extract the data
            features : list/str, optional (default="all")
                The name/names of features to fetch. By default, all available
                dynamic features are returned.
            st : Optional (default=None)
                start time from where to fetch the data.
            en : Optional (default=None)
                end time untill where to fetch the data
            as_dataframe : bool, optional (default=False)
                if true, the returned data is pandas DataFrame otherwise it
                is xarray dataset

        Examples
        --------
            >>> from water_datasets import RainfallRunoff
            >>> camels = RainfallRunoff('CAMELS_AUS')
            >>> camels.fetch_dynamic_features('224214A', as_dataframe=True).unstack()
            >>> camels.dynamic_features
            >>> camels.fetch_dynamic_features('224214A',
            ... features=['tmax_AWAP', 'vprp_AWAP', 'streamflow_mmd'],
            ... as_dataframe=True).unstack()
        """
        return self.dataset.fetch_dynamic_features(stn_id, features, st, en, as_dataframe)


    def fetch_station_features(
            self,
            station: str,
            dynamic_features: Union[str, list, None] = 'all',
            static_features: Union[str, list, None] = None,
            as_ts: bool = False,
            st: Union[str, None] = None,
            en: Union[str, None] = None,
            **kwargs
    ) -> pd.DataFrame:
        """
        Fetches features for one station.

        Parameters
        -----------
            station :
                station id/gauge id for which the data is to be fetched.
            dynamic_features : str/list, optional
                names of dynamic features/attributes to fetch
            static_features :
                names of static features/attributes to be fetches
            as_ts : bool
                whether static features are to be converted into a time
                series or not. If yes then the returned time series will be of
                same length as that of dynamic attribtues.
            st : str,optional
                starting point from which the data to be fetched. By default,
                the data will be fetched from where it is available.
            en : str, optional
                end point of data to be fetched. By default the dat will be fetched

        Returns
        -------
        pd.DataFrame
            dataframe if as_ts is True else it returns a dictionary of static and
            dynamic features for a station/gauge_id

        Examples
        --------
            >>> from water_datasets import RainfallRunoff
            >>> dataset = RainfallRunoff('CAMELS_AUS')
            >>> dataset.fetch_station_features('912101A')

        """
        return self.dataset.fetch_station_features(station, dynamic_features, static_features, as_ts, st, en, **kwargs)

    def plot_stations(
            self,
            stations:List[str] = None,
            marker='.',
            ax:plt_Axes = None,
            show:bool = True,
            **kwargs
    )->plt_Axes:
        """
        plots coordinates of stations

        Parameters
        ----------
        stations :
            name/names of stations. If not given, all stations will be plotted
        marker :
            marker to use.
        ax : plt.Axes
            matplotlib axes to draw the plot. If not given, then
            new axes will be created.
        show : bool
        **kwargs

        Returns
        -------
        plt.Axes

        Examples
        --------
        >>> from water_datasets import RainfallRunoff
        >>> dataset = RainfallRunoff('CAMELS_AUS')
        >>> dataset.plot_stations()
        >>> dataset.plot_stations(['1', '2', '3'])
        >>> dataset.plot_stations(marker='o', ms=0.3)
        >>> ax = dataset.plot_stations(marker='o', ms=0.3, show=False)
        >>> ax.set_title("Stations")
        >>> plt.show()

        """
        return self.dataset.plot_stations(stations, marker, ax, show, **kwargs)

    def q_mmd(
            self,
            stations: Union[str, List[str]] = None
    )->pd.DataFrame:
        """
        returns streamflow in the units of milimeter per day. This is obtained
        by diving ``q``/area

        parameters
        ----------
        stations : str/list
            name/names of stations. Default is None, which will return
            area of all stations

        Returns
        --------
        pd.DataFrame
            a pandas DataFrame whose indices are time-steps and columns
            are catchment/station ids.

        """
        return self.dataset.q_mmd(stations)

    def stn_coords(
            self,
            stations:Union[str, List[str]] = None
    ) ->pd.DataFrame:
        """
        returns coordinates of stations as DataFrame
        with ``long`` and ``lat`` as columns.

        Parameters
        ----------
        stations :
            name/names of stations. If not given, coordinates
            of all stations will be returned.

        Returns
        -------
        coords :
            pandas DataFrame with ``long`` and ``lat`` columns.
            The length of dataframe will be equal to number of stations
            wholse coordinates are to be fetched.

        Examples
        --------
        >>> from water_datasets import RainfallRunoff
        >>> dataset = RainfallRunoff('CAMELS_CH')
        >>> dataset.stn_coords() # returns coordinates of all stations
        >>> dataset.stn_coords('2004')  # returns coordinates of station whose id is 2004
        >>> dataset.stn_coords(['2004', '6004'])  # returns coordinates of two stations

        >>> from water_datasets import RainfallRunoff
        >>> dataset = RainfallRunoff('CAMELS_AUS')
        >>> dataset.stn_coords() # returns coordinates of all stations
        >>> dataset.stn_coords('912101A')  # returns coordinates of station whose id is 912101A
        >>> dataset.stn_coords(['G0050115', '912101A'])  # returns coordinates of two stations

        """
        return self.dataset.stn_coords(stations)

    def get_boundary(
            self,
            catchment_id: str,
            as_type: str = 'numpy'
    ):
        """
        returns boundary of a catchment in a required format

        Parameters
        ----------
        catchment_id : str
            name/id of catchment
        as_type : str
            'numpy' or 'geopandas'
        
        Examples
        --------
        >>> from water_datasets import RainfallRunoff
        >>> dataset = RainfallRunoff('CAMELS_SE')
        >>> dataset.get_boundary(dataset.stations()[0])
        """
        return self.dataset.get_boundary(catchment_id, as_type)

    def plot_catchment(
            self,
            catchment_id: str,
            ax: plt_Axes = None,
            show: bool = True,
            **kwargs
    )->plt.Axes:
        """
        plots catchment boundaries

        Parameters
        ----------
        ax : plt.Axes
            matplotlib axes to draw the plot. If not given, then
            new axes will be created.
        show : bool
        **kwargs

        Returns
        -------
        plt.Axes

        Examples
        --------
        >>> from water_datasets import RainfallRunoff
        >>> dataset = RainfallRunoff('CAMELS_AUS')
        >>> dataset.plot_catchment()
        >>> dataset.plot_catchment(marker='o', ms=0.3)
        >>> ax = dataset.plot_catchment(marker='o', ms=0.3, show=False)
        >>> ax.set_title("Catchment Boundaries")
        >>> plt.show()

        """
        return self.dataset.plot_catchment(catchment_id, ax, show, **kwargs)

    def stations(self)->List[str]:
        """
        returns names of all stations

        Examples
        --------
        >>> from water_datasets import RainfallRunoff
        >>> dataset = RainfallRunoff('CAMELS_AUS')
        >>> dataset.stations()
        """
        return self.dataset.stations()