#ifndef GWMODEL_H
#define GWMODEL_H

#ifndef GWMODEL_SHARED_LIB

#include "gwmodelpp/spatialweight/CGwmBandwidthWeight.h"
#include "gwmodelpp/spatialweight/CGwmCRSDistance.h"
#include "gwmodelpp/spatialweight/CGwmDistance.h"
#include "gwmodelpp/spatialweight/CGwmDMatDistance.h"
#include "gwmodelpp/spatialweight/CGwmMinkwoskiDistance.h"
#include "gwmodelpp/spatialweight/CGwmSpatialWeight.h"
#include "gwmodelpp/spatialweight/CGwmWeight.h"

#include "gwmodelpp/CGwmAlgorithm.h"
#include "gwmodelpp/CGwmBandwidthSelector.h"
#include "gwmodelpp/CGwmVariableForwardSelector.h"
#include "gwmodelpp/CGwmSimpleLayer.h"
#include "gwmodelpp/CGwmSpatialAlgorithm.h"
#include "gwmodelpp/CGwmSpatialMonoscaleAlgorithm.h"
#include "gwmodelpp/CGwmGWRBase.h"
#include "gwmodelpp/CGwmGWRBasic.h"

#else   // GWMODEL_SHARED_LIB

#ifdef WIN32
#ifdef CREATE_DLL_EXPORTS
#define GWMODEL_API __declspec(dllexport)
#else
#define GWMODEL_API __declspec(dllimport)
#endif 
#else
#define GWMODEL_API  
#endif

#include "gwmodelpp/GwmRegressionDiagnostic.h"

enum KernelFunctionType
{
    Gaussian,
    Exponential,
    Bisquare,
    Tricube,
    Boxcar
};

enum BandwidthSelectionCriterionType
{
    AIC,
    CV
};

struct GwmVariable;
struct IGwmRegressionAnalysis;
class CGwmDistance;
class CGwmWeight;
class CGwmSpatialWeight;
class CGwmSimpleLayer;
class CGwmAlgorithm;
class CGwmSpatialAlgorithm;
class CGwmSpatialMonoscaleAlgorithm;
class CGwmGWRBase;
class CGwmGWRBasic;

struct GwmMatInterface
{
    unsigned long long rows;
    unsigned long long cols;
    double* data;
};

struct GwmStringInterface
{
    const char* str;
};

struct GwmStringListInterface
{
    size_t size;
    GwmStringInterface* items;
};

struct GwmVariableInterface
{
    int index;
    bool isNumeric;
    const char* name;
};

struct GwmVariableListInterface
{
    size_t size;
    GwmVariableInterface* items;
};

struct GwmBandwidthKernelInterface
{
    double size;
    bool isAdaptive;
    KernelFunctionType type;
};

extern "C" GWMODEL_API CGwmDistance* gwmodel_create_crs_distance(bool isGeogrphical);
extern "C" GWMODEL_API CGwmWeight* gwmodel_create_bandwidth_weight(double size, bool isAdaptive, KernelFunctionType type);
extern "C" GWMODEL_API CGwmSpatialWeight* gwmodel_create_spatial_weight(CGwmDistance* distance, CGwmWeight* weight);
extern "C" GWMODEL_API CGwmSimpleLayer* gwmodel_create_simple_layer(GwmMatInterface pointsInterface, GwmMatInterface dataInterface, GwmStringListInterface fieldsInterface);
extern "C" GWMODEL_API CGwmSpatialAlgorithm* gwmodel_create_algorithm();
extern "C" GWMODEL_API CGwmGWRBasic* gwmodel_create_gwr_algorithm();

extern "C" GWMODEL_API void gwmodel_delete_crs_distance(CGwmDistance* instance);
extern "C" GWMODEL_API void gwmodel_delete_bandwidth_weight(CGwmWeight* instance);
extern "C" GWMODEL_API void gwmodel_delete_spatial_weight(CGwmSpatialWeight* instance);
extern "C" GWMODEL_API void gwmodel_delete_simple_layer(CGwmSimpleLayer* instance);
extern "C" GWMODEL_API void gwmodel_delete_algorithm(CGwmSpatialAlgorithm* instance);
extern "C" GWMODEL_API void gwmodel_delete_gwr_algorithm(CGwmGWRBasic* instance);

extern "C" GWMODEL_API void gwmodel_set_gwr_source_layer(CGwmGWRBasic* algorithm, CGwmSimpleLayer* layer);
extern "C" GWMODEL_API void gwmodel_set_gwr_spatial_weight(CGwmGWRBasic* algorithm, CGwmSpatialWeight* spatial);
extern "C" GWMODEL_API void gwmodel_set_gwr_dependent_variable(CGwmGWRBasic* regression, GwmVariableInterface depVar);
extern "C" GWMODEL_API void gwmodel_set_gwr_independent_variable(CGwmGWRBasic* regression, GwmVariableListInterface indepVarList);
extern "C" GWMODEL_API void gwmodel_set_gwr_predict_layer(CGwmGWRBasic* algorithm, CGwmSimpleLayer* layer);
extern "C" GWMODEL_API void gwmodel_set_gwr_bandwidth_autoselection(CGwmGWRBasic* algorithm, BandwidthSelectionCriterionType criterion);
extern "C" GWMODEL_API void gwmodel_set_gwr_indep_vars_autoselection(CGwmGWRBasic* algorithm, double threshold);
extern "C" GWMODEL_API void gwmodel_set_gwr_options(CGwmGWRBasic* algorithm, bool hasHatMatrix);

extern "C" GWMODEL_API void gwmodel_run_gwr(CGwmGWRBasic* algorithm);

extern "C" GWMODEL_API void gwmodel_get_simple_layer_points(CGwmSimpleLayer* layer, GwmMatInterface* pointsInterface);
extern "C" GWMODEL_API void gwmodel_get_simple_layer_data(CGwmSimpleLayer* layer, GwmMatInterface* dataInterface);
extern "C" GWMODEL_API void gwmodel_get_simple_layer_fields(CGwmSimpleLayer* layer, GwmStringListInterface* fieldsInterface);
extern "C" GWMODEL_API void gwmodel_get_gwr_spatial_weight(CGwmGWRBasic* gwr, CGwmDistance** distance, CGwmWeight** weight);
extern "C" GWMODEL_API void gwmodel_get_gwr_result_layer(CGwmGWRBasic* gwr, CGwmSimpleLayer** resultLayer);
extern "C" GWMODEL_API void gwmodel_get_gwr_coefficients(CGwmGWRBasic* gwr, GwmMatInterface* coefficientsInterface);
extern "C" GWMODEL_API GwmRegressionDiagnostic gwmodel_get_gwr_diagnostic(CGwmGWRBasic* gwr);

extern "C" GWMODEL_API bool gwmodel_as_bandwidth_weight(CGwmWeight* weight, GwmBandwidthKernelInterface* bandwidth);

#endif

#endif  // GWMODEL_H