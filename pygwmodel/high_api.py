import numpy as np
import pandas as pd
import geopandas as gp
from shapely.geometry import Point
from typing import List, Union
from enum import Enum
from .pygwmodel import CySimpleLayer, CyVariable, CyVariableList
from .pygwmodel import CyDistance, CyCRSDistance
from .pygwmodel import CyWeight, CyBandwidthWeight
from .pygwmodel import CyGWRBasic, CyGWSS, CyGWPCA


class KernelType(Enum):
    GAUSSIAN = 0

class BandwidthSelectionCriterionType(Enum):
    AIC = 0
    CV = 1


def sdf_to_layer(sdf: gp.GeoDataFrame, variables: List[str]):
    """
    Convert GeoDataFrame to SimpleLayer
    """
    cyg_coord = np.asfortranarray([[x, y] for x, y in zip(sdf.centroid.x, sdf.centroid.y)])
    cyg_data = np.asfortranarray(sdf[variables])
    cyg_fields = [x.encode() for x in variables]
    return CySimpleLayer(cyg_coord, cyg_data, cyg_fields)


def layer_to_sdf(layer: CySimpleLayer, geometry: gp.GeoSeries=None):
    """
    Convert GeoDataFrame to SimpleLayer
    """
    result_geom = layer.points  # type: np.ndarray
    result_attr = layer.data
    result_field = [f.decode() for f in layer.fields]
    result_data = {f: result_attr[:, i] for i, f in enumerate(result_field)}
    return gp.GeoDataFrame({
        **result_data,
        "geometry": gp.GeoSeries([Point(result_geom[p, 0], result_geom[p, 1]) for p in range(result_geom.shape[0])])
    }) if geometry is None else gp.GeoDataFrame({
        **result_data,
        "geometry": geometry
    })


class GWRBasic:
    """
    Basic GWR python high api class.
    """

    def __init__(self, sdf: gp.GeoDataFrame, depen_var: str, indep_vars: List[str], bw: float=None, adaptive: bool=True, kernel: KernelType=KernelType.GAUSSIAN, longlat: bool=True):
        """
        docstring
        """
        if not isinstance(sdf, gp.GeoDataFrame):
            raise ValueError("sdf must be a GeoDataFrame")
        self.sdf = sdf
        self.depen_var = depen_var
        self.indep_vars = indep_vars
        self.bw = bw
        self.kernel = kernel
        self.adaptive = adaptive
        self.longlat = longlat
        self.result_layer = None
        self.diagnostic = None
        self.bandwidth_select_criterions = None
        self.indep_var_select_criterions = None
    
    def fit(self, hatmatrix: bool=True, optimize_bw: BandwidthSelectionCriterionType=None, optimize_var: float=None, multithreads: int=None):
        """
        Run algorithm and return result
        """
        ''' Extract data
        '''
        cyg_data_layer = sdf_to_layer(self.sdf, [self.depen_var] + self.indep_vars)
        cyg_distance = CyCRSDistance(self.longlat)
        cyg_weight = CyBandwidthWeight(self.bw, self.adaptive, self.kernel.value)
        cyg_depen_var = CyVariable(0, True, self.depen_var.encode("utf-8"))
        cyg_indep_vars = CyVariableList([CyVariable(i + 1, True, n.encode("utf-8")) for i, n in enumerate(self.indep_vars)])
        ''' Create cython GWR
        '''
        cyg_gwr_basic = CyGWRBasic(cyg_data_layer, cyg_depen_var, cyg_indep_vars, cyg_weight, cyg_distance, hatmatrix)
        if self.bw is None and optimize_bw is None:
            optimize_bw = BandwidthSelectionCriterionType.CV
        if optimize_bw is not None:
            if optimize_bw == BandwidthSelectionCriterionType.AIC or optimize_bw == BandwidthSelectionCriterionType.CV:
                cyg_gwr_basic.enable_bandwidth_autoselection(optimize_bw.value)
            else:
                raise ValueError("optimize_bw must be BandwidthSelectionCriterionType.AIC(0) or BandwidthSelectionCriterionType.CV(1)")
        if optimize_var is not None:
            if isinstance(optimize_var, float) and optimize_var > 0:
                cyg_gwr_basic.enable_indep_var_autoselection(optimize_var)
            else:
                raise ValueError("optimize_var must be a positive real number")
        if multithreads is not None:
            if isinstance(multithreads, int) and multithreads > 0:
                cyg_gwr_basic.enable_openmp(multithreads)
            else:
                raise ValueError("multithreads must be a positive integer")
        cyg_gwr_basic.run()
        if self.bw is None or optimize_bw is not None:
            self.bw = cyg_gwr_basic.bandwidth()
            self.bandwidth_select_criterions = cyg_gwr_basic.bandwidth_select_criterions()
        if optimize_var is not None:
            self.indep_vars = [v.decode() for v in cyg_gwr_basic.indep_vars()]
            self.indep_var_select_criterions = cyg_gwr_basic.indep_var_select_criterions()
        ''' Get result layer
        '''
        self.result_layer = layer_to_sdf(cyg_gwr_basic.result_layer, self.sdf.geometry)
        if hatmatrix:
            self.diagnostic = cyg_gwr_basic.diagnostic()
        return self

    def predict(self, targets: gp.GeoDataFrame, multithreads: int=None):
        """
        Predict
        """
        if self.bw is None:
            raise ValueError("Bandwidth cannot be None when predicting")
        ''' Extract data
        '''
        cyg_source_layer = sdf_to_layer(self.sdf, [self.depen_var] + self.indep_vars)
        cyg_predict_layer = sdf_to_layer(targets, self.indep_vars)
        cyg_distance = CyCRSDistance(self.longlat)
        cyg_weight = CyBandwidthWeight(self.bw, self.adaptive, self.kernel.value)
        cyg_depen_var = CyVariable(0, True, self.depen_var.encode("utf-8"))
        cyg_indep_vars = CyVariableList([CyVariable(i + 1, True, n.encode("utf-8")) for i, n in enumerate(self.indep_vars)])
        ''' Create cython GWR
        '''
        cyg_gwr_basic = CyGWRBasic(cyg_source_layer, cyg_depen_var, cyg_indep_vars, cyg_weight, cyg_distance, False)
        cyg_gwr_basic.set_predict_layer(cyg_predict_layer)
        if multithreads is not None:
            if isinstance(multithreads, int) and multithreads > 0:
                cyg_gwr_basic.enable_openmp(multithreads)
            else:
                raise ValueError("multithreads must be a positive integer")
        ''' Get result layer
        '''
        cyg_gwr_basic.run()
        return layer_to_sdf(cyg_gwr_basic.result_layer, targets.geometry)


class GWSS:
    """
    GWSS python high api class.
    """

    def __init__(self, sdf: gp.GeoDataFrame, variables: List[str], bw: float, adaptive: bool=True, kernel: KernelType=KernelType.GAUSSIAN, longlat: bool=True, quantile: bool=False, first_only: bool=False):
        """
        docstring
        """
        if not isinstance(sdf, gp.GeoDataFrame):
            raise ValueError("sdf must be a GeoDataFrame")
        self.sdf = sdf
        self.variables = variables
        self.bw = bw
        self.kernel = kernel
        self.adaptive = adaptive
        self.longlat = longlat
        self.result_layer = None
        self.quantile = quantile
        self.first_only = first_only

    def fit(self, multithreads: int=None):
        """
        Run algorithm and return result
        """
        ''' Extract data
        '''
        cyg_data_layer = sdf_to_layer(self.sdf, self.variables)
        cyg_distance = CyCRSDistance(self.longlat)
        cyg_weight = CyBandwidthWeight(self.bw, self.adaptive, self.kernel.value)
        cyg_in_vars = CyVariableList([CyVariable(i, True, n.encode("utf-8")) for i, n in enumerate(self.variables)])
        ''' Create cython GWSS
        '''
        cyg_gwss = CyGWSS(cyg_data_layer, cyg_in_vars, cyg_weight, cyg_distance, self.quantile, self.first_only)
        if multithreads is not None:
            if isinstance(multithreads, int) and multithreads > 0:
                cyg_gwss.enable_openmp(multithreads)
            else:
                raise ValueError("multithreads must be a positive integer")
        cyg_gwss.run()
        self.result_layer = layer_to_sdf(cyg_gwss.result_layer, self.sdf.geometry)
        return self


class GWPCA:
    """
    GWPCA python high api class.
    """
    result_layer = None
    loadings = None
    local_pv = None

    def __init__(self, sdf: gp.GeoDataFrame, variables: List[str], bw: float, adaptive: bool=True, kernel: KernelType=KernelType.GAUSSIAN, longlat: bool=True, keepComponents: int=2):
        """
        docstring
        """
        if not isinstance(sdf, gp.GeoDataFrame):
            raise ValueError("sdf must be a GeoDataFrame")
        self.sdf = sdf
        self.variables = variables
        self.bw = bw
        self.kernel = kernel
        self.adaptive = adaptive
        self.longlat = longlat
        self.keepComponents = keepComponents

    def fit(self):
        """
        Run algorithm and return result
        """
        ''' Extract data
        '''
        cyg_data_layer = sdf_to_layer(self.sdf, self.variables)
        cyg_distance = CyCRSDistance(self.longlat)
        cyg_weight = CyBandwidthWeight(self.bw, self.adaptive, self.kernel.value)
        # cyg_spatial_weight = cyg_sw.SpatialWeight(cyg_distance, cyg_weight)
        cyg_in_vars = CyVariableList([CyVariable(i, True, n.encode("utf-8")) for i, n in enumerate(self.variables)])
        ''' Create cython GWPCA
        '''
        cyg_gwpca = CyGWPCA(cyg_data_layer, cyg_in_vars, cyg_weight, cyg_distance, self.keepComponents)
        cyg_gwpca.run()
        self.result_layer = layer_to_sdf(cyg_gwpca.result_layer, self.sdf.geometry)
        ''' Get loadings
        '''
        self.local_pv = cyg_gwpca.local_pv()
        self.loadings = cyg_gwpca.loadings()
        return self
