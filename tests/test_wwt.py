import os
import site

# add the parent directory in the path
wd_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
site.addsitedir(wd_dir)

import unittest

from water_datasets import ec_removal_biochar, mg_photodegradation

class TestWWT(unittest.TestCase):

    def test_qe_biochar(self):
        data, _ = ec_removal_biochar()
        assert data.shape == (3757, 27)
        data, encoders = ec_removal_biochar(encoding="le")
        assert data.shape == (3757, 27)
        assert data.sum().sum() >= 10346311.47
        adsorbents = encoders['adsorbent'].inverse_transform(data.iloc[:, 22])
        assert len(set(adsorbents)) == 15
        pollutants = encoders['pollutant'].inverse_transform(data.iloc[:, 23])
        assert len(set(pollutants)) == 14
        ww_types = encoders['ww_type'].inverse_transform(data.iloc[:, 24])
        assert len(set(ww_types)) == 4
        adsorption_types = encoders['adsorption_type'].inverse_transform(data.iloc[:, 25])
        assert len(set(adsorption_types)) == 2
        data, encoders = ec_removal_biochar(encoding="ohe")
        assert data.shape == (3757, 58)
        adsorbents = encoders['adsorbent'].inverse_transform(data.iloc[:, 22:37].values)
        assert len(set(adsorbents)) == 15
        pollutants =  encoders['pollutant'].inverse_transform(data.iloc[:, 37:51].values)
        assert len(set(pollutants)) == 14
        ww_types = encoders['ww_type'].inverse_transform(data.iloc[:, 51:55].values)
        assert len(set(ww_types)) == 4
        adsorption_types = encoders['adsorption_type'].inverse_transform(data.iloc[:, -3:-1].values)
        assert len(set(adsorption_types)) == 2
        return

    def test_mg_photodegradation(self):
        data, *_ = mg_photodegradation()
        assert data.shape == (1200, 12)
        data, cat_enc, an_enc = mg_photodegradation(encoding="le")
        assert data.shape == (1200, 12)
        assert data.sum().sum() >= 406354.95
        cat_enc.inverse_transform(data.iloc[:, 9].values.astype(int))
        an_enc.inverse_transform(data.iloc[:, 10].values.astype(int))
        data, cat_enc, an_enc = mg_photodegradation(encoding="ohe")
        assert data.shape == (1200, 31)
        cat_enc.inverse_transform(data.iloc[:, 9:24].values)
        an_enc.inverse_transform(data.iloc[:, 24:30].values)

        return



if __name__ == '__main__':
    unittest.main()