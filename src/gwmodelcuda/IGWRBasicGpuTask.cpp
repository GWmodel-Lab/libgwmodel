#include "IGWRBasicGpuTask.h"
#include "GWRBasicGPUTask.h"


IGWRBasicGpuTask * GWRBasicGpuTaskFit_Create(int nDp, int nVar, int distanceType)
{
	return new GWRBasicGpuTask(nDp, nVar, (gwm::Distance::DistanceType)distanceType);
}

IGWRBasicGpuTask * GWRBasicGpuTaskPredict_Create(int nDp, int nVar, int distanceType, int nPredictPoints)
{
	return new GWRBasicGpuTask(nDp, nVar, (gwm::Distance::DistanceType)distanceType, nPredictPoints);
}

void GWRBasicGpuTask_Del(IGWRBasicGpuTask * pInstance)
{
	delete pInstance;
}
