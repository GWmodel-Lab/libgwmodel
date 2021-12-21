import sys
import unittest
import numpy as np
import pandas as pd
import geopandas as gp
from pygwmodel import GWRBasic


class TestGWRBasic(unittest.TestCase):
    
    def setUp(self):
        londonhp_csv = pd.read_csv(sys.argv[1])
        self.londonhp = gp.GeoDataFrame(londonhp_csv, geometry=gp.points_from_xy(londonhp_csv.x, londonhp_csv.y))

    def test_minimal(self):
        londonhp_depen = 'PURCHASE'
        londonhp_indep = ["FLOORSZ", "UNEMPLOY", "PROF"]
        algorithm = GWRBasic(self.londonhp, londonhp_depen, londonhp_indep, 36.0, longlat=False).fit()

        diagnostic0 = np.array([
            2436.60445730413,
            2448.27206524754,
            0.708010632044736,
            0.674975341723766
        ])
        diagnostic = np.array([
            algorithm.diagnostic['AIC'],
            algorithm.diagnostic['AICc'],
            algorithm.diagnostic['RSquare'],
            algorithm.diagnostic['RSquareAdjust']
        ])
        self.assertTrue(np.all(np.abs(diagnostic0 - diagnostic) < 1e-8))
    

if __name__ == '__main__':
    unittest.main(argv=[''], exit=False, verbosity=2)
