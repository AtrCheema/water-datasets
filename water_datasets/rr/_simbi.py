
from .camels import Camels

class Simbi(Camels):
    """
    monthly rainfall from 1905 - 2005, daily rainfall from 1920-1940, 70 daily
    streamflow series, and 23 monthly temperature series for 24 catchments of Haiti
    """
    url = "https://dataverse.ird.fr/dataset.xhtml?persistentId=doi:10.23708/02POK6"

    def __init__(
            self,
            path: str = None,
            to_netcdf:bool = True,
            overwrite:bool = False,
            verbosity:int = 1,
            **kwargs
    ):
        """
        Arguments:
            path: path where the Simbi dataset has been downloaded. This path
                must contain five zip files and one xlsx file. If None, then the
                data will be downloaded.
            to_netcdf :
        """
        super().__init__(path=path, verbosity=verbosity, **kwargs)    

        self._download()