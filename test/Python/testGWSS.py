import sys
import numpy as np
import pandas as pd
import geopandas as gp
from pygwmodel import GWSS

londonhp_csv = pd.read_csv(sys.argv[1])
londonhp = gp.GeoDataFrame(londonhp_csv, geometry=gp.points_from_xy(londonhp_csv.x, londonhp_csv.y))
londonhp_vars = ["PURCHASE", "FLOORSZ", "UNEMPLOY", "PROF"]
londonhp_gwss = GWSS(londonhp, londonhp_vars, 36.0, longlat=True)
londonhp_gwss_result: gp.GeoDataFrame = londonhp_gwss.fit().result_layer

result = pd.DataFrame(londonhp_gwss_result).drop('geometry', axis=1)
result_q = result.apply(lambda x: np.quantile(x, [0, 0.25, 0.5, 0.75, 1], interpolation='midpoint'), axis=0)

localmean_q0 = np.array([[155530.887621432, 71.3459254279447, 6.92671958853926, 39.0446823327541],
    [163797.287583358, 73.3206754261603, 7.53173813461806, 40.4678577700236],
    [174449.84375947, 74.1174325820277, 7.99672839037902, 43.2051994175928],
    [183893.664229323, 75.3118600659781, 8.59668519607066, 45.2164679493302],
    [188967.723827491, 77.0911277060738, 8.95571485750978, 47.5614366837457]])
localmean_q = result_q.loc[:, 'PURCHASE_LM':'PROF_LM'].values
if np.all(np.abs(localmean_q - localmean_q0) < 1e-8):
    exit(0)
