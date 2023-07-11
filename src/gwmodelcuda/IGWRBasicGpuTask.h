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
	virtual void setDp(int i, double u, double v) = 0;
	virtual void setRp(int i, double u, double v) = 0;
	virtual void setDmat(int i, int j, double value) = 0;

	virtual double betas(int i, int k) = 0;
	virtual double betasSE(int i, int k) = 0;
	virtual double shat1() = 0;
	virtual double shat2() = 0;
	virtual double qDiag(int i) = 0;


	virtual bool fit(
		bool hatmatrix,
		double p, double theta, bool longlat,
		double bw, int kernel, bool adaptive,
		int groupl, int gpuID
	) = 0;

	virtual bool predict(
		double p, double theta, bool longlat,
		double bw, int kernel, bool adaptive,
		int groupl, int gpuID
	) = 0;

	virtual double cv(
		double p, double theta, bool longlat,
		double bw, int kernel, bool adaptive,
		int groupl, int gpuID
	) = 0;
};

extern "C" GWMODELCUDA_API IGWRBasicGpuTask* GpuTask_Create(int N, int K, bool rp_given, int n, bool dm_given);
extern "C" GWMODELCUDA_API void GpuTask_Del(IGWRBasicGpuTask* pInstance);

#endif  // IGWMCUDA
