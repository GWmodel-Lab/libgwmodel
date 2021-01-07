from cbase cimport CGwmGWSS
from cbase cimport gwmodel_create_gwss_algorithm
from cbase cimport gwmodel_set_gwss_source_layer
from cbase cimport gwmodel_set_gwss_spatial_weight
from cbase cimport gwmodel_set_gwss_variables
from cbase cimport gwmodel_set_gwss_options
from cbase cimport gwmodel_delete_gwss_algorithm
from cbase cimport gwmodel_get_gwss_result_layer
from cbase cimport gwmodel_get_simple_layer_points
from cbase cimport gwmodel_get_simple_layer_data
from cbase cimport gwmodel_get_simple_layer_fields
from cbase cimport gwmodel_get_gwss_local_mean
from cbase cimport gwmodel_get_gwss_local_sdev
from cbase cimport gwmodel_get_gwss_local_var
from cbase cimport gwmodel_get_gwss_local_skew
from cbase cimport gwmodel_get_gwss_local_cv
from cbase cimport gwmodel_get_gwss_local_cov
from cbase cimport gwmodel_get_gwss_local_corr
from cbase cimport gwmodel_get_gwss_local_spearman_rho
from cbase cimport gwmodel_get_gwss_local_median
from cbase cimport gwmodel_get_gwss_local_iqr
from cbase cimport gwmodel_get_gwss_local_qi
from cbase cimport gwmodel_set_gwss_openmp
from cbase cimport gwmodel_run_gwss
    
cdef class GWSS:
    cdef CGwmGWSS* _c_instance