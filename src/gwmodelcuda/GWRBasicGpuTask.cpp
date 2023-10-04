#include "GWRBasicGpuTask.h"

#include "gwmodel.h"
#include "StdTelegram.h"

using namespace std;
using namespace arma;
using namespace gwm;

bool GWRBasicGpuTask::fit(bool hasIntercept)
{
    SpatialWeight sw(mWeight, mDistance);
    GWRBasic algorithm(mX, mY, mCoords, sw, true, hasIntercept);
    algorithm.setIsAutoselectBandwidth(mIsOptimizeBandwidth);
    algorithm.setBandwidthSelectionCriterion(mBandwidthOptimizationCriterion);
    algorithm.setIsAutoselectIndepVars(mIsOptimizeVariables);
    algorithm.setIndepVarSelectionThreshold(mOptimizeVariablesThreshold);
    algorithm.setParallelType(ParallelType::CUDA);
    algorithm.setTelegram(make_unique<StdTelegram>());
    try
    {
        algorithm.fit();
        mBetas = algorithm.betas();
        mBetasSE = algorithm.betasSE();
        mSHat = algorithm.sHat();
        mQDiag = algorithm.qDiag();
        mS = algorithm.s();
        mDiagnostic = algorithm.diagnostic();
        
        if (mIsOptimizeBandwidth)
        {
            mOptimizedBandwidth = algorithm.spatialWeight().weight<BandwidthWeight>()->bandwidth();
        }

        if (mIsOptimizeVariables)
        {
            mSelectedVars = algorithm.selectedVariables();
            mVariableOptimizationCriterionList = algorithm.indepVarsSelectionCriterionList();
        }
    }
    catch(const std::exception& e)
    {
        return false;
    }

    return true;
}

bool GWRBasicGpuTask::predict(bool hasIntercept)
{
    SpatialWeight sw(mWeight, mDistance);
    GWRBasic algorithm(mX, mY, mCoords, sw, true, hasIntercept);
    algorithm.setParallelType(ParallelType::CUDA);
    algorithm.setTelegram(make_unique<StdTelegram>());
    try
    {
        algorithm.predict(mPredictLocations);
        mBetas = algorithm.betas();
    }
    catch(const std::exception& e)
    {
        return false;
    }

    return true;
}
