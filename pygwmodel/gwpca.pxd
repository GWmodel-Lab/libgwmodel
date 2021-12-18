from .cbase cimport CGwmGWPCA
from .cbase cimport gwmodel_create_gwpca_algorithm
from .cbase cimport gwmodel_delete_gwpca_algorithm
from .cbase cimport gwmodel_get_gwpca_result_layer
from .cbase cimport gwmodel_get_gwpca_local_pv
from .cbase cimport gwmodel_get_gwpca_loadings
from .cbase cimport gwmodel_get_gwpca_sdev
from .cbase cimport gwmodel_get_gwpca_scores
from .cbase cimport gwmodel_set_gwpca_source_layer
from .cbase cimport gwmodel_set_gwpca_variables
from .cbase cimport gwmodel_set_gwpca_spatial_weight
from .cbase cimport gwmodel_set_gwpca_options
from .cbase cimport gwmodel_run_gwpca
from .cbase cimport gwmodel_get_simple_layer_points
from .cbase cimport gwmodel_get_simple_layer_data
from .cbase cimport gwmodel_get_simple_layer_fields
    
cdef class GWPCA:
    cdef CGwmGWPCA* _c_instance