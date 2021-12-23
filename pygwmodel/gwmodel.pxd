from libcpp.string cimport string
from libcpp.vector cimport vector

cdef extern from "armadillo" namespace "arma":
    cdef cppclass mat:
        mat()
        mat(int n_rows, int n_cols) except +
        mat(double * aux_mem, int n_rows, int n_cols) except +
        int n_rows
        int n_cols
        int n_elem
        double* memptr()
    
    cdef cppclass cube:
        cube()
        cube(int n_row, int n_cols, int n_slices) except +
        cube(double* aux_mem, int n_row, int n_cols, int n_slices) except +
        int n_rows
        int n_cols
        int n_elem
        int n_slices
        double* memptr()
        mat slice(unsigned long long slice_number)

cdef extern from "CGwmSimpleLayer.h":
    cdef cppclass CGwmSimpleLayer:
        CGwmSimpleLayer() except +
        CGwmSimpleLayer(const mat& points, const mat& data, const vector[string]& fields) except +
        mat points() const;
        mat data() const;
        vector[string] fields() const;
        unsigned long long featureCount() const;

cdef extern from "GwmVariable.h":
    cdef cppclass GwmVariable:
        GwmVariable()
        GwmVariable(int i, bint numeric, string n)
        int index
        bint isNumeric
        string name

cdef extern from "spatialweight/CGwmDistance.h":
    cdef cppclass CGwmDistance:
        CGwmDistance()

cdef extern from "spatialweight/CGwmCRSDistance.h":
    cdef cppclass CGwmCRSDistance(CGwmDistance):
        CGwmCRSDistance()
        CGwmCRSDistance(bint isGeographic)
        bint geographic() const
        void setGeographic(bint geographic)

cdef extern from "spatialweight/CGwmWeight.h":
    cdef cppclass CGwmWeight:
        CGwmWeight()

cdef extern from "spatialweight/CGwmBandwidthWeight.h":
    cdef cppclass CGwmBandwidthWeight(CGwmWeight):
        enum KernelFunctionType:
            Gaussian = 0
            Exponential = 1
            Bisquare = 2
            Tricube = 3
            Boxcar = 4
        CGwmBandwidthWeight()
        CGwmBandwidthWeight(double size, bint adaptive, KernelFunctionType kernel)
        double bandwidth() const
        void setBandwidth(double bandwidth)
        bint adaptive() const
        void setAdaptive(bint adaptive)
        KernelFunctionType kernel() const
        void setKernel(const KernelFunctionType &kernel)

cdef extern from "spatialweight/CGwmSpatialWeight.h":
    cdef cppclass CGwmSpatialWeight:
        CGwmSpatialWeight()
        CGwmSpatialWeight(CGwmWeight* weight, CGwmDistance* distance)
        CGwmWeight *weight() const
        void setWeight(CGwmWeight *weight)
        CGwmDistance *distance() const
        void setDistance(CGwmDistance *distance)


cdef extern from "CGwmAlgorithm.h":
    cdef cppclass CGwmAlgorithm:
        CGwmAlgorithm()
        bint isValid()
        void run()


cdef extern from "CGwmSpatialAlgorithm.h":
    cdef cppclass CGwmSpatialAlgorithm(CGwmAlgorithm):
        CGwmSpatialAlgorithm()
        CGwmSimpleLayer* sourceLayer()
        void setSourceLayer(CGwmSimpleLayer* layer)
        void setSourceLayer(const CGwmSimpleLayer& layer)
        CGwmSimpleLayer* resultLayer() const
        void setResultLayer(CGwmSimpleLayer* layer);


cdef extern from "CGwmSpatialMonoscaleAlgorithm.h":
    cdef cppclass CGwmSpatialMonoscaleAlgorithm(CGwmSpatialAlgorithm):
        CGwmSpatialMonoscaleAlgorithm()
        CGwmSpatialWeight spatialWeight() const
        void setSpatialWeight(const CGwmSpatialWeight &spatialWeight)


cdef extern from "IGwmMultivariableAnalysis.h":
    cdef cppclass IGwmMultivariableAnalysis:
        vector[GwmVariable] variables() const
        void setVariables(const vector[GwmVariable]& variables)


cdef extern from "CGwmGWSS.h":
    cdef cppclass CGwmGWSS(CGwmSpatialMonoscaleAlgorithm, IGwmMultivariableAnalysis):
        CGwmGWSS()
        bint quantile() const;
        void setQuantile(bint quantile);
        bint isCorrWithFirstOnly() const;
        void setIsCorrWithFirstOnly(bint corrWithFirstOnly);
        mat localMean() const;
        mat localSDev() const;
        mat localSkewness() const;
        mat localCV() const;
        mat localVar() const;
        mat localMedian() const;
        mat iqr() const;
        mat qi() const;
        mat localCov() const;
        mat localCorr() const;
        mat localSCorr() const;


cdef extern from "CGwmGWPCA.h":
    cdef cppclass CGwmGWPCA(CGwmSpatialMonoscaleAlgorithm, IGwmMultivariableAnalysis):
        CGwmGWPCA()
        int keepComponents();
        void setKeepComponents(int k);
        mat localPV()
        mat sdev()
        cube loadings();
        cube scores();

        