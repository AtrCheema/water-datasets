Rainfall Runoff
***************


.. list-table:: Stations per Source
   :widths: 25 20 20 30
   :header-rows: 1

   * - Source Name
     - Number of Daily Stations
     - Number of Hourly Stations
     - Reference
   * - CAMELS_AUS
     - 222, 561
     - 
     - `Flower et al., 2021 <https://doi.org/10.5194/essd-13-3847-2021>`_
   * - CAMELS_GB
     - 671
     - 
     - `Coxon et al., 2020 <https://doi.org/10.5194/hess-24-4877-2020>`_
   * - CAMELS_BR
     - 897
     - 
     - `Chagas et al., 2020 <https://doi.org/10.5194/essd-12-2075-2020>`_
   * - CAMELS_US
     - 671
     - 
     - `Newman et al., 2014 <https://gdex.ucar.edu/dataset/camels.html>`_
   * - CAMELS_CL
     - 516
     - 
     - `Alvarez-Garreton et al., 2018 <https://doi.org/10.5194/hess-22-5817-2018>`_
   * - CAMELS_DK
     - 304
     - 
     - `Liu et al., 2024 <https://doi.org/10.5194/essd-2024-292>`_
   * - CAMELS_CH
     - 331
     - 
     - `Hoege et al., 2023 <https://doi.org/10.5194/essd-15-5755-2023>`_
   * - CAMELS_DE
     - 1555
     - 
     - `Loritz et al., 2024 <https://essd.copernicus.org/preprints/essd-2024-318/>`_
   * - CAMELS_SE
     - 50
     -
     - `Teutschbein et al., 2024 <https://doi.org/10.1002/gdj3.239>`_
   * - LamaH
     - 859
     -
     - `Klingler et al., 2021 <https://doi.org/10.5194/essd-13-4529-2021>`_
   * - LamaHIce
     - 111
     -
     - `Helgason and Nijssen 2024 <https://doi.org/10.5194/essd-16-2741-2024>`_
   * - HYSETS
     - 14425
     -
     - `Arsenault et al., 2020 <https://doi.org/10.1038/s41597-020-00583-2>`_
   * - GRDCCaravan
     - 5357
     -
     - `Faerber et al., 2023 <https://zenodo.org/records/10074416>`_
   * - Bull
     - 484
     -
     - `Aparicio et al., 2024 <https://doi.org/10.1038/s41597-024-03594-5>`_    
   * - WaterBenchIowa
     - 125
     -
     - `Demir et al., 2022 <https://doi.org/10.5194/essd-14-5605-2022>`_
   * - CCAM
     - 111
     -
     - `Hao et al., 2021 <https://doi.org/10.5194/essd-13-5591-2021>`_
   * - RRLuleaSweden
     - 1
     -
     - `Broekhuizen et al., 2020 <https://doi.org/10.5194/hess-24-869-2020>`_
   * - CABra
     - 735
     - 
     - `Almagro et al., 2021 <https://doi.org/10.5194/hess-25-3105-2021>`_ 
   * - HYPE
     - 561
     - 
     - `Arciniega-Esparza and Birkel, 2020 <https://zenodo.org/records/4029572>`_
   * - Simbi
     - 24
     -
     - `Bathelemy et al., 2024 <doi: 10.5194/essd-16-2073-2024>`_
   * - CAMELS_IND
     - 472
     -
     - `Mangukiya et al., 2024 <https://doi.org/10.5194/essd-2024-379>`_


High Level API
==============
It consists of a unified interface to access all the datasets. The datasets are accessed by their names.

.. autoclass:: water_datasets.rr.RainfallRunoff
   :members:
   :undoc-members:
   :show-inheritance:

   .. automethod:: __init__


Low Level API
=============
The datasets can be accessed individually by their names.


.. autoclass:: water_datasets.rr.Camels
   :members:
   :undoc-members:
   :show-inheritance:

   .. automethod:: __init__


.. autoclass:: water_datasets.rr.CAMELS_AUS
   :members:
   :show-inheritance:

   .. automethod:: __init__


.. autoclass:: water_datasets.rr.CAMELS_GB
   :members:
   :undoc-members:
   :show-inheritance:

   .. automethod:: __init__


.. autoclass:: water_datasets.rr.CAMELS_BR
   :members:
   :show-inheritance:

   .. automethod:: __init__


.. autoclass:: water_datasets.rr.CAMELS_US
   :members:
   :show-inheritance:

   .. automethod:: __init__


.. autoclass:: water_datasets.rr.CAMELS_CL
   :members:
   :show-inheritance:

   .. automethod:: __init__


.. autoclass:: water_datasets.rr.CAMELS_DK
   :members:
   :show-inheritance:

   .. automethod:: __init__


.. autoclass:: water_datasets.rr.LamaH
   :members:
   :show-inheritance:

   .. automethod:: __init__


.. autoclass:: water_datasets.rr.LamaHIce
   :members:
   :show-inheritance:

   .. automethod:: __init__


.. autoclass:: water_datasets.rr.HYSETS
   :members:
   :show-inheritance:

   .. automethod:: __init__


.. autoclass:: water_datasets.rr.HYPE
   :members:
   :show-inheritance:

   .. automethod:: __init__


.. autoclass:: water_datasets.rr.CCAM
   :members:
   :show-inheritance:

   .. automethod:: __init__


.. autoclass:: water_datasets.rr.CABra
   :members:
   :show-inheritance:

   .. automethod:: __init__


.. autoclass:: water_datasets.rr.CAMELS_CH
   :members:
   :show-inheritance:

   .. automethod:: __init__


.. autoclass:: water_datasets.rr.CAMELS_DE
   :members:
   :show-inheritance:

   .. automethod:: __init__


.. autoclass:: water_datasets.rr.GRDCCaravan
   :members:
   :show-inheritance:

   .. automethod:: __init__


.. autoclass:: water_datasets.rr.CAMELS_SE
   :members:
   :show-inheritance:

   .. automethod:: __init__


.. autoclass:: water_datasets.rr.WaterBenchIowa
   :members:
   :show-inheritance:

   .. automethod:: __init__


.. autoclass:: water_datasets.rr.Simbi
   :members:
   :show-inheritance:

   .. automethod:: __init__


.. autoclass:: water_datasets.RRLuleaSweden
   :members:
   :show-inheritance:

   .. automethod:: __init__


.. autoclass:: water_datasets.Bull
   :members:
   :show-inheritance:

   .. automethod:: __init__

.. autoclass:: water_datasets.CAMELS_IND
   :members:
   :show-inheritance:

   .. automethod:: __init__

.. autoclass:: water_datasets.rr._denmark.CAMELS_DK
   :members:
   :show-inheritance:

   .. automethod:: __init__