#ifndef IGWMCUDA
#define IGWMCUDA

#ifdef WIN32
#ifdef CREATDLL_EXPORTS
#define GWMODELCUDA_API __declspec(dllexport)
#else
#define GWMODELCUDA_API __declspec(dllimport)
#endif // CREATDLL_EXPORTS
#else
#define GWMODELCUDA_API
#endif

class GWMODELCUDA_API IGWRBasicGpuTask 
{
public:
	virtual void setX(int i, int k, double value) = 0;
	virtual void setY(int i, double value) = 0;
	virtual void setCoords(int i, double u, double v) = 0;
	virtual void setPredictLocations(int i, double u, double v) = 0;

	virtual void setDistanceType(int type) = 0;
	virtual void setCRSDistanceGergraphic(bool isGeographic) = 0;
	virtual void setMinkwoskiDistancePoly(int poly) = 0;
	virtual void setMinkwoskiDistanceTheta(double theta) = 0;
	
	virtual void setBandwidthSize(double bw) = 0;
	virtual void setBandwidthAdaptive(bool adaptive) = 0;
	virtual void setBandwidthKernel(int kernel) = 0;

	virtual void enableBandwidthOptimization(int criterion) = 0;
	virtual void enableVariablesOptimization(double threshold) = 0;

	virtual double betas(int i, int k) = 0;
	virtual double betasSE(int i, int k) = 0;
	virtual double shat1() = 0;
	virtual double shat2() = 0;
	virtual double qDiag(int i) = 0;
	virtual unsigned long long sRows() = 0;
	virtual double s(int i, int k) = 0;

	virtual double diagnosticRSS() = 0;
	virtual double diagnosticAIC() = 0;
	virtual double diagnosticAICc() = 0;
	virtual double diagnosticENP() = 0;
	virtual double diagnosticEDF() = 0;
	virtual double diagnosticRSquare() = 0;
	virtual double diagnosticRSquareAdjust() = 0;

	virtual double optimizedBandwidth()  = 0;
	virtual unsigned long long selectedVarSize()  = 0;
	virtual unsigned long long selectedVar(unsigned long long i)  = 0;
    virtual unsigned long long variableSelectionCriterionSize() = 0;
    virtual unsigned long long variableSelectionCriterionItemVarSize(unsigned long long i) = 0;
    virtual unsigned long long variableSelectionCriterionItemVar(unsigned long long i, unsigned long long j) = 0;
    virtual double variableSelectionCriterionItemValue(unsigned long long i) = 0;

	virtual bool fit(bool hasIntercept) = 0;

	virtual bool predict(bool hasIntercept) = 0;
};

extern "C" GWMODELCUDA_API IGWRBasicGpuTask* GWRBasicGpuTaskFit_Create(int nDp, int nVar, int distanceType);
extern "C" GWMODELCUDA_API IGWRBasicGpuTask* GWRBasicGpuTaskPredict_Create(int nDp, int nVar, int distanceType, int nPredictPoints);
extern "C" GWMODELCUDA_API void GWRBasicGpuTask_Del(IGWRBasicGpuTask * pInstance);

#endif  // IGWMCUDA
