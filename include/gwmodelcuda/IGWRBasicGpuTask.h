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
	virtual std::size_t sRows() = 0;
	virtual double s(int i, int k) = 0;


	virtual bool fit(bool hasIntercept) = 0;

	virtual bool predict(bool hasIntercept) = 0;
};

extern "C" GWMODELCUDA_API IGWRBasicGpuTask* GpuTask_Create(int nDp, int nVar, int distanceType);
extern "C" GWMODELCUDA_API void GpuTask_Del(IGWRBasicGpuTask * pInstance);

#endif  // IGWMCUDA
