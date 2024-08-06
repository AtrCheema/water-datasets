
import os
from typing import Union, Tuple, Any, List

import numpy as np
import pandas as pd

from .utils import encode_column, LabelEncoder, OneHotEncoder


def mg_photodegradation(
        inputs: list = None,
        target: str = "Efficiency (%)",
        encoding:str = None
)->Tuple[pd.DataFrame,
         Union[LabelEncoder, OneHotEncoder, Any],
         Union[LabelEncoder, OneHotEncoder, Any]]:
    """
    This data is about photocatalytic degradation of melachite green dye using
    nobel metal dobe BiFeO3. For further description of this data see
    `Jafari et al., 2023 <https://doi.org/10.1016/j.jhazmat.2022.130031>`_ and
    for the use of this data for removal efficiency prediction `see <https://github.com/ZeeshanHJ/Photocatalytic_Performance_Prediction>`_ .
    This dataset consists of 1200 points collected during ~135 experiments.

    Parameters
    ----------
        inputs : list, optional
            features to use as input. By default following features are used as input

                - ``Catalyst_type``
                - ``Surface area``
                - ``Pore Volume``
                - ``Catalyst_loading (g/L)``
                - ``Light_intensity (W)``
                - ``time (min)``
                - ``solution_pH``
                - ``HA (mg/L)``
                - ``Anions``
                - ``Ci (mg/L)``
                - ``Cf (mg/L)``

        target : str, optional, default="Efficiency (%)"
            features to use as target. By default ``Efficiency (%)`` is used as target
            which is photodegradation removal efficiency of dye from wastewater. Following
            are valid target names

                - ``Efficiency (%)``
                - ``k_first``
                - ``k_2nd``

        encoding : str, default=None
            type of encoding to use for the two categorical features i.e., ``Catalyst_type``
            and ``Anions``, to convert them into numberical. Available options are ``ohe``,
            ``le`` and None. If ohe is selected the original input columns are replaced
            with ohe hot encoded columns. This will result in 6 columns for Anions and
            15 columns for Catalyst_type.

    Returns
    -------
    data : pd.DataFrame
        a pandas dataframe consisting of input and output features. The default
        setting will result in dataframe shape of (1200, 12)
    cat_encoder :
        catalyst encoder
    an_encoder :
        encoder for anions

    Examples
    --------
    >>> from ai4water.datasets import mg_photodegradation
    >>> mg_data, catalyst_encoder, anion_encoder = mg_photodegradation()
    >>> mg_data.shape
    (1200, 12)
    ... # the default encoding is None, but if we want to use one hot encoder
    >>> mg_data_ohe, cat_enc, an_enc = mg_photodegradation(encoding="ohe")
    >>> mg_data_ohe.shape
    (1200, 31)
    >>> cat_enc.inverse_transform(mg_data_ohe.iloc[:, 9:24].values)
    >>> an_enc.inverse_transform(mg_data_ohe.iloc[:, 24:30].values)
    ... # if we want to use label encoder
    >>> mg_data_le, cat_enc, an_enc = mg_photodegradation(encoding="le")
    >>> mg_data_le.shape
    (1200, 12)
    >>> cat_enc.inverse_transform(mg_data_le.iloc[:, 9].values.astype(int))
    >>> an_enc.inverse_transform(mg_data_le.iloc[:, 10].values.astype(int))
    ... # By default the target is efficiency but if we want
    ... # to use first order k as target
    >>> mg_data_k, _, _ = mg_photodegradation(target="k_first")
    ... # if we want to use 2nd order k as target
    >>> mg_data_k2, _, _ = mg_photodegradation(target="k_2nd")

    """

    df = pd.read_csv(
    "https://raw.githubusercontent.com/ZeeshanHJ/Photocatalytic_Performance_Prediction/main/Raw%20data.csv"
    )
    default_inputs = ['Surface area', 'Pore Volume', 'Catalyst_loading (g/L)',
                      'Light_intensity (W)', 'time (min)', 'solution_pH', 'HA (mg/L)',
                      'Ci (mg/L)', 'Cf (mg/L)', 'Catalyst_type', 'Anions',
                      ]
    default_targets = ['Efficiency (%)', 'k_first', 'k_2nd']

    # first order
    df["k_first"] = np.log(df["Ci (mg/L)"] / df["Cf (mg/L)"]) / df["time (min)"]

    # k second order
    df["k_2nd"] = ((1 / df["Cf (mg/L)"]) - (1 / df["Ci (mg/L)"])) / df["time (min)"]

    if inputs is None:
        inputs = default_inputs

    if not isinstance(target, list):
        if isinstance(target, str):
            target = [target]
    elif isinstance(target, list):
        pass
    else:
        target = default_targets

    assert isinstance(target, list)

    assert all(trgt in default_targets for trgt in target)

    df = df[inputs + target]

    # consider encoding of categorical features
    cat_encoder, an_encoder = None, None
    if encoding:
        df, cols_added, cat_encoder = encode_column(df, "Catalyst_type", encoding)
        df, an_added, an_encoder = encode_column(df, "Anions", encoding)

        # move the target to the end
        for t in target:
            df[t] = df.pop(t)

    return df, cat_encoder, an_encoder


def ec_removal_biochar(
        input_features:List[str]=None,
        encoding:str = None
)->Tuple[pd.DataFrame, dict]:
    """
    Data of removal of emerging pollutants from wastewater
    using biochar. The data consists of three types of features,
    1) adsorption experimental conditions, 2) elemental composition of
    adsorbent (biochar) and parameters representing
    physical and synthesis conditions of biochar.
    For more description of this data see `Jaffari et al., 2023 <https://doi.org/10.1016/j.cej.2023.143073>`_


    Parameters
    ----------
    input_features :
        By default following features are used as input
            - ``Adsorbent``
            - ``Pyrolysis temperature``
            - ``Pyrolysis time``
            - ``C``
            - ``H``
            - ``O``
            - ``N``
            - ``(O+N)/C``
            - ``Ash``
            - ``H/C``
            - ``O/C``
            - ``Surface area``
            - ``Pore volume``
            - ``Average pore size``
            - ``Pollutant``
            - ``Adsorption time``
            - ``concentration``
            - ``Solution pH``
            - ``RPM``
            - ``Volume``
            - ``Adsorbent dosage``
            - ``Adsorption temperature``
            - ``Ion concentration``
            - ``Humid acid``
            - ``Wastewater type``
            - ``Adsorption type``

    encoding : str, default=None
        the type of encoding to use for categorical features. If not None, it should
        be either ``ohe`` or ``le``.

    Returns
    --------
    tuple
        A tuple of length two. The first element is a DataFrame while the
        second element is a dictionary consisting of encoders with ``adsorbent``
        ``pollutant``, ``ww_type`` and ``adsorption_type`` as keys.

    Examples
    --------
    >>> from ai4water.datasets import ec_removal_biochar
    >>> data, *_ = ec_removal_biochar()
    >>> data.shape
    (3757, 27)
    >>> data, encoders = ec_removal_biochar(encoding="le")
    >>> data.shape
    (3757, 27)
    >>> len(set(encoders['adsorbent'].inverse_transform(data.iloc[:, 22])))
    15
    >>> len(set(encoders['pollutant'].inverse_transform(data.iloc[:, 23])))
    14
    >>> set(encoders['ww_type'].inverse_transform(data.iloc[:, 24]))
    {'Ground water', 'Lake water', 'Secondary effluent', 'Synthetic'}
    >>> set(encoders['adsorption_type'].inverse_transform(data.iloc[:, 25]))
    {'Competative', 'Single'}

    We can also use one hot encoding to convert categorical features into
    numerical features. This will obviously increase the number of features/columns in DataFrame

    >>> data, encoders = ec_removal_biochar(encoding="ohe")
    >>> data.shape
    (3757, 58)
    >>> len(set(encoders['adsorption_type'].inverse_transform(data.iloc[:, 22:37].values)))
    15
    >>> len(set(encoders['pollutant'].inverse_transform(data.iloc[:, 37:51].values)))
    14
    >>> set(encoders['ww_type'].inverse_transform(data.iloc[:, 51:55].values))
    {'Ground water', 'Lake water', 'Secondary effluent', 'Synthetic'}
    >>> set(encoders['adsorption_type'].inverse_transform(data.iloc[:, 55:-1].values))
    {'Competative', 'Single'}

    """
    fpath = os.path.join(os.path.dirname(__file__), "data", 'qe_biochar_ec.csv')
    url = 'https://raw.githubusercontent.com/ZeeshanHJ/Adsorption-capacity-prediction-for-ECs/main/Raw_data.csv'

    if os.path.exists(fpath):
        data = pd.read_csv(fpath)
    else:
        data = pd.read_csv(url)
        # remove space in 'Pyrolysis temperature '
        data['Pyrolysis temperature'] = data.pop('Pyrolysis temperature ')

        data['Adsorbent'] = data.pop('Adsorbent')
        data['Pollutant'] = data.pop('Pollutant')
        data['Wastewater type'] = data.pop('Wastewater type')
        data['Adsorption type'] = data.pop('Adsorption type')

        data['Capacity'] = data.pop('Capacity')

        data.to_csv(fpath, index=False)

    def_inputs = [
        'Pyrolysis temperature',
        'Pyrolysis time',
        'C',
        'H',
        'O',
        'N',
        '(O+N)/C',
        'Ash',
        'H/C',
        'O/C',
        'Surface area',
        'Pore volume',
        'Average pore size',
        'Adsorption time',
        'Initial concentration',
        'Solution pH',
        'RPM',
        'Volume',
        'Adsorbent dosage',
        'Adsorption temperature',
        'Ion concentration',
        'Humic acid',
        'Adsorbent',
        'Pollutant',
        'Wastewater type',
        'Adsorption type',
    ]

    if input_features is not None:
        assert isinstance(input_features, list)
        assert all([feature in def_inputs for feature in input_features])
    else:
        input_features = def_inputs

    data = data[input_features + ['Capacity']]

    ads_enc, pol_enc, wwt_enc, adspt_enc = None, None, None, None
    if encoding:
        data, _, ads_enc = encode_column(data, 'Adsorbent', encoding)
        data, _, pol_enc = encode_column(data, 'Pollutant', encoding)
        data, _, wwt_enc = encode_column(data, 'Wastewater type', encoding)
        data, _, adspt_enc = encode_column(data, 'Adsorption type', encoding)

        # putting capacity at the end
        data['Capacity'] = data.pop('Capacity')

    encoders = {
        "adsorbent": ads_enc,
        "pollutant": pol_enc,
        "ww_type": wwt_enc,
        "adsorption_type": adspt_enc
    }
    return data, encoders
