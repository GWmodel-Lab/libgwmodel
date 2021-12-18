import sys
import numpy as np
import pandas as pd
import geopandas as gp
from pygwmodel import GWPCA

londonhp_csv = pd.read_csv(sys.argv[1])
londonhp = gp.GeoDataFrame(londonhp_csv, geometry=gp.points_from_xy(londonhp_csv.x, londonhp_csv.y))
londonhp_vars = ["FLOORSZ", "UNEMPLOY", "PROF"]
algorithm = GWPCA(londonhp, londonhp_vars, 36.0, longlat=True, keepComponents=2).fit()
algorithm_result: gp.GeoDataFrame = algorithm.result_layer
algorithm_loadings: np.ndarray = algorithm.loadings

result = pd.DataFrame(algorithm_result).drop('geometry', axis=1)
result_q = result.apply(lambda x: np.quantile(x, [0, 0.25, 0.5, 0.75, 1], interpolation='midpoint'), axis=0)

comp_q0 = np.array([
    [86.09381920388, 7.38948790899526],
    [87.2417310474256, 10.0805823313445],
    [88.5114946422145, 11.4166428700704],
    [89.8514496001622, 12.6890545321313],
    [92.5449003124064, 13.8382823156345]
])
comp_q = result_q.loc[:, 'Comp.1_PV':'Comp.2_PV'].values  # type: ignore
if not np.all(np.abs(comp_q0 - comp_q) < 1e-8):
    print("testPythonGWPCA: local principle value is not equal.")
    print("Turth:", comp_q0, sep='\n')
    print("Result:", comp_q, sep='\n')
    exit(1)

loadings_q0 = np.array([
    [
        [0.997738665168933,-0.0115292388648388,-0.040450830035712],
        [0.998673840689833,-0.00822122467004101,-0.00468318323510321],
        [0.999297415085303,-0.00389424492785976,0.0320948265474252],
        [0.999678999646886,0.00274831974093292,0.0508510246498129],
        [0.999999194544384,0.0105326992413149,0.0662213367046487]
    ],
    [
        [-0.0671714511032381,-0.219162658504117,-0.97924877722135],
        [-0.0513759501838017,-0.214853304247932,0.976144875457391],
        [-0.0321827960857794,-0.211329933955831,0.976665314794129],
        [0.00517581158157478,-0.204353440937033,0.978464099165948],
        [0.0417635544237787,0.202661857194208,0.980384688418526]
    ]
])
loadings_q = np.apply_along_axis(lambda x: np.quantile(x, [0, 0.25, 0.5, 0.75, 1], interpolation='midpoint'), axis=1, arr=algorithm_loadings)
if not np.all(np.abs(loadings_q0 - loadings_q) < 1e-8):
    print("testPythonGWSS: loadings is not equal.")
    print("Turth:", loadings_q0, sep='\n')
    print("Result:", loadings_q, sep='\n')
    exit(1)

exit(0)
