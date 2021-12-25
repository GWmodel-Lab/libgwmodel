import sys
import unittest
import numpy as np
import pandas as pd
import geopandas as gp
from pygwmodel import GWRBasic
import pygwmodel


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
    
    def test_minimal_multithreads(self):
        londonhp_depen = 'PURCHASE'
        londonhp_indep = ["FLOORSZ", "UNEMPLOY", "PROF"]
        algorithm = GWRBasic(self.londonhp, londonhp_depen, londonhp_indep, 36.0, longlat=False).fit(multithreads=8)

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

    def test_autoselect_bandwidth(self):
        londonhp_depen = 'PURCHASE'
        londonhp_indep = ["FLOORSZ", "UNEMPLOY", "PROF"]
        algorithm = GWRBasic(self.londonhp, londonhp_depen, londonhp_indep, 36.0, longlat=False).fit(optimize_bw=pygwmodel.BandwidthSelectionCriterionType.CV)
        self.assertEqual(algorithm.bw, 67)

    def test_autoselect_bandwidth_multithread(self):
        londonhp_depen = 'PURCHASE'
        londonhp_indep = ["FLOORSZ", "UNEMPLOY", "PROF"]
        algorithm = GWRBasic(self.londonhp, londonhp_depen, londonhp_indep, 36.0, longlat=False).fit(optimize_bw=pygwmodel.BandwidthSelectionCriterionType.CV, multithreads=8)
        self.assertEqual(algorithm.bw, 67)

    def test_autoselect_indepvars(self):
        londonhp_depen = 'PURCHASE'
        londonhp_indep = ["FLOORSZ", "UNEMPLOY", "PROF"]
        algorithm = GWRBasic(self.londonhp, londonhp_depen, londonhp_indep, 36.0, longlat=False).fit(optimize_var=3.0)
        criterion = algorithm.indep_var_select_criterions
        self.assertSequenceEqual(criterion[0][0], ['UNEMPLOY'])
        self.assertSequenceEqual(criterion[1][0], ['PROF'])
        self.assertSequenceEqual(criterion[2][0], ['FLOORSZ'])
        self.assertSequenceEqual(criterion[3][0], ['FLOORSZ', 'PROF'])
        self.assertSequenceEqual(criterion[4][0], ['FLOORSZ', 'UNEMPLOY'])
        self.assertSequenceEqual(criterion[5][0], ['FLOORSZ', 'UNEMPLOY', 'PROF'])
        self.assertAlmostEqual(criterion[0][1], 2551.61359020599, delta=1e-8)
        self.assertAlmostEqual(criterion[1][1], 2551.30032201349, delta=1e-8)
        self.assertAlmostEqual(criterion[2][1], 2468.93236280013, delta=1e-8)
        self.assertAlmostEqual(criterion[3][1], 2452.86447942033, delta=1e-8)
        self.assertAlmostEqual(criterion[4][1], 2450.59642666509, delta=1e-8)
        self.assertAlmostEqual(criterion[5][1], 2452.80388934625, delta=1e-8)
        self.assertSequenceEqual(algorithm.indep_vars, ['FLOORSZ', 'PROF'])

    def test_autoselect_indepvars_multithreads(self):
        londonhp_depen = 'PURCHASE'
        londonhp_indep = ["FLOORSZ", "UNEMPLOY", "PROF"]
        algorithm = GWRBasic(self.londonhp, londonhp_depen, londonhp_indep, 36.0, longlat=False).fit(optimize_var=3.0, multithreads=8)
        criterion = algorithm.indep_var_select_criterions
        self.assertSequenceEqual(criterion[0][0], ['UNEMPLOY'])
        self.assertSequenceEqual(criterion[1][0], ['PROF'])
        self.assertSequenceEqual(criterion[2][0], ['FLOORSZ'])
        self.assertSequenceEqual(criterion[3][0], ['FLOORSZ', 'PROF'])
        self.assertSequenceEqual(criterion[4][0], ['FLOORSZ', 'UNEMPLOY'])
        self.assertSequenceEqual(criterion[5][0], ['FLOORSZ', 'UNEMPLOY', 'PROF'])
        self.assertAlmostEqual(criterion[0][1], 2551.61359020599, delta=1e-8)
        self.assertAlmostEqual(criterion[1][1], 2551.30032201349, delta=1e-8)
        self.assertAlmostEqual(criterion[2][1], 2468.93236280013, delta=1e-8)
        self.assertAlmostEqual(criterion[3][1], 2452.86447942033, delta=1e-8)
        self.assertAlmostEqual(criterion[4][1], 2450.59642666509, delta=1e-8)
        self.assertAlmostEqual(criterion[5][1], 2452.80388934625, delta=1e-8)
        self.assertSequenceEqual(algorithm.indep_vars, ['FLOORSZ', 'PROF'])

    def test_autoselect_all(self):
        londonhp_depen = 'PURCHASE'
        londonhp_indep = ["FLOORSZ", "UNEMPLOY", "PROF"]
        algorithm = GWRBasic(self.londonhp, londonhp_depen, londonhp_indep, 36.0, longlat=False).fit(optimize_var=3.0, optimize_bw=pygwmodel.BandwidthSelectionCriterionType.CV)
        criterion = algorithm.indep_var_select_criterions
        self.assertSequenceEqual(criterion[0][0], ['UNEMPLOY'])
        self.assertSequenceEqual(criterion[1][0], ['PROF'])
        self.assertSequenceEqual(criterion[2][0], ['FLOORSZ'])
        self.assertSequenceEqual(criterion[3][0], ['FLOORSZ', 'PROF'])
        self.assertSequenceEqual(criterion[4][0], ['FLOORSZ', 'UNEMPLOY'])
        self.assertSequenceEqual(criterion[5][0], ['FLOORSZ', 'UNEMPLOY', 'PROF'])
        self.assertAlmostEqual(criterion[0][1], 2551.61359020599, delta=1e-8)
        self.assertAlmostEqual(criterion[1][1], 2551.30032201349, delta=1e-8)
        self.assertAlmostEqual(criterion[2][1], 2468.93236280013, delta=1e-8)
        self.assertAlmostEqual(criterion[3][1], 2452.86447942033, delta=1e-8)
        self.assertAlmostEqual(criterion[4][1], 2450.59642666509, delta=1e-8)
        self.assertAlmostEqual(criterion[5][1], 2452.80388934625, delta=1e-8)
        self.assertSequenceEqual(algorithm.indep_vars, ['FLOORSZ', 'PROF'])
        self.assertEqual(algorithm.bw, 31)

    def test_autoselect_all_multithreads(self):
        londonhp_depen = 'PURCHASE'
        londonhp_indep = ["FLOORSZ", "UNEMPLOY", "PROF"]
        algorithm = GWRBasic(self.londonhp, londonhp_depen, londonhp_indep, 36.0, longlat=False).fit(optimize_var=3.0, optimize_bw=pygwmodel.BandwidthSelectionCriterionType.CV, multithreads=8)
        criterion = algorithm.indep_var_select_criterions
        self.assertSequenceEqual(criterion[0][0], ['UNEMPLOY'])
        self.assertSequenceEqual(criterion[1][0], ['PROF'])
        self.assertSequenceEqual(criterion[2][0], ['FLOORSZ'])
        self.assertSequenceEqual(criterion[3][0], ['FLOORSZ', 'PROF'])
        self.assertSequenceEqual(criterion[4][0], ['FLOORSZ', 'UNEMPLOY'])
        self.assertSequenceEqual(criterion[5][0], ['FLOORSZ', 'UNEMPLOY', 'PROF'])
        self.assertAlmostEqual(criterion[0][1], 2551.61359020599, delta=1e-8)
        self.assertAlmostEqual(criterion[1][1], 2551.30032201349, delta=1e-8)
        self.assertAlmostEqual(criterion[2][1], 2468.93236280013, delta=1e-8)
        self.assertAlmostEqual(criterion[3][1], 2452.86447942033, delta=1e-8)
        self.assertAlmostEqual(criterion[4][1], 2450.59642666509, delta=1e-8)
        self.assertAlmostEqual(criterion[5][1], 2452.80388934625, delta=1e-8)
        self.assertSequenceEqual(algorithm.indep_vars, ['FLOORSZ', 'PROF'])
        self.assertEqual(algorithm.bw, 31)


if __name__ == '__main__':
    unittest.main(argv=[''], verbosity=2)
