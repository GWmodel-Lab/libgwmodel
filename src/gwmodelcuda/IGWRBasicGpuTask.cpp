#include "IGWRBasicGpuTask.h"
#include "GWRBasicGPUTask.h"


IGWRBasicGpuTask * GpuTask_Create(int nDp, int nVar, int distanceType)
{
	return new GWRBasicGpuTask(nDp, nVar, (gwm::Distance::DistanceType)distanceType);
}

void GpuTask_Del(IGWRBasicGpuTask * pInstance)
{
	delete pInstance;
}
