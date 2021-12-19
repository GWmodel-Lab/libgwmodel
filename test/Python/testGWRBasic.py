import sys
import numpy as np
import pandas as pd
import geopandas as gp
from pygwmodel import GWRBasic

londonhp_csv = pd.read_csv(sys.argv[1])
londonhp = gp.GeoDataFrame(londonhp_csv, geometry=gp.points_from_xy(londonhp_csv.x, londonhp_csv.y))
londonhp_depen = 'PURCHASE'
londonhp_indep = ["FLOORSZ", "UNEMPLOY", "PROF"]
algorithm = GWRBasic(londonhp, londonhp_depen, londonhp_indep, 36.0, longlat=False).fit()

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
if not np.all(np.abs(diagnostic0 - diagnostic) < 1e-8):
    print("testPythonGWRBasic: diagnostic is not equal.")
    print("Turth:", diagnostic0, sep='\n')
    print("Result:", diagnostic, sep='\n')
    exit(1)

exit(0)
