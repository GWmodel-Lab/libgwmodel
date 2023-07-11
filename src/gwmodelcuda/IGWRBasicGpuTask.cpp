#include "IGWRBasicGWRTask.h"
#include "GWRBasicGPUTask.h"


IGWRBasicGpuTask * CreateGWRBasicGpuTask(int N, int K, bool rp_given, int n, bool dm_given)
{
	return new GWRBasicGpuTask(N, K, rp_given, n, dm_given);
}

void DeleteGWRBasicGpuTask(IGWRBasicGpuTask * pInstance)
{
	delete pInstance;
}

bool RunGWRBasicGpuTask(IGWRBasicGpuTask * pInstance, bool hatmatrix, double p, double theta, bool longlat, double bw, int kernel, bool adaptive, int groupl, int gpuID)
{
	return pInstance->fit(hatmatrix, p, theta, longlat, bw, kernel, adaptive, groupl, gpuID);
}