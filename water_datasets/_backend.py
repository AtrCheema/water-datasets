
__all__ = ['netCDF4', 'plt', 'shapefile', 'xarray', 'matplotlib']

try:
    import netCDF4
except (ImportError, ModuleNotFoundError):
    netCDF4 = None

try:
    import matplotlib.pyplot as plt
    import matplotlib
except (ModuleNotFoundError, ImportError) as e:
    matplotlib, plt = None, None

try:
    import shapefile
except (ModuleNotFoundError, ImportError) as e:
    shapefile = None


try:
    import xarray
except (ModuleNotFoundError, ImportError) as e:
    xarray = None