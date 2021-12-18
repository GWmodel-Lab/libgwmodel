from .gwpca cimport *
from .simple_layer cimport SimpleLayer, CGwmSimpleLayer
from .spatial_weight cimport Distance, Weight, SpatialWeight
from .mat_interface cimport MatInterface, mat2numpy, mat2interface
from .variable_interface cimport VariableListInterface
from .name_list_interface cimport NameListInterface, names2list
from cython.view cimport array as cvarray

cdef class GWPCA:
    def __cinit__(self, SimpleLayer source_layer, SpatialWeight spatial_weight, VariableListInterface variables, unsigned int keepComponents):
        self._c_instance = gwmodel_create_gwpca_algorithm()
        gwmodel_set_gwpca_source_layer(self._c_instance, source_layer._c_instance)
        gwmodel_set_gwpca_spatial_weight(self._c_instance, spatial_weight._c_instance)
        gwmodel_set_gwpca_variables(self._c_instance, variables._c_instance)
        gwmodel_set_gwpca_options(self._c_instance, keepComponents)
    
    def __dealloc__(self):
        gwmodel_delete_gwpca_algorithm(self._c_instance)
    
    @property
    def result_layer(self):
        cdef CGwmSimpleLayer* layer = gwmodel_get_gwpca_result_layer(self._c_instance)
        cdef MatInterface points = mat2interface(gwmodel_get_simple_layer_points(layer))
        cdef MatInterface data = mat2interface(gwmodel_get_simple_layer_data(layer))
        return SimpleLayer(points, data, NameListInterface(names2list(gwmodel_get_simple_layer_fields(layer))))
    
    def local_pv(self):
        return mat2numpy(gwmodel_get_gwpca_local_pv(self._c_instance))
    
    def loadings(self):
        return mat2numpy(gwmodel_get_gwpca_loadings(self._c_instance))

    def sdev(self):
        return mat2numpy(gwmodel_get_gwpca_sdev(self._c_instance))

    def run(self):
        gwmodel_run_gwpca(self._c_instance)