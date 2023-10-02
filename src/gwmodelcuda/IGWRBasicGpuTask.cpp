#include "IGWRBasicGpuTask.h"
#include "GWRBasicGPUTask.h"


IGWRBasicGpuTask * CreateGWRBasicGpuFitTask(int nDp, int nVar, int distanceType)
{
	return new GWRBasicGpuTask(nDp, nVar, (gwm::Distance::DistanceType)distanceType);
}

void DeleteGWRBasicGpuTask(IGWRBasicGpuTask * pInstance)
{
	delete pInstance;
}
