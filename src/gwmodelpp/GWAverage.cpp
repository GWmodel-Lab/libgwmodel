#include "GWAverage.h"
#include <assert.h>
#include <map>

#ifdef ENABLE_OPENMP
#include <omp.h>
#endif

using namespace std;
using namespace arma;
using namespace gwm;

vec GWAverage::del(vec x, uword rowcount){
    vec res;
    if (rowcount == 0)
        res = x.rows(rowcount + 1, x.n_rows - 1);
    else if (rowcount == x.n_rows - 1)
        res = x.rows(0,x.n_rows-2);
    else
        res = join_cols(x.rows(0,rowcount - 1),x.rows(rowcount + 1, x.n_rows - 1));
    return res;
}

vec GWAverage::findq(const mat &x, const vec &w)
{
    uword lw = w.n_rows;
    uword lp = 3;
    vec q = vec(lp,fill::zeros);
    vec xo = sort(x);
    vec wo = w(sort_index(x));
    vec Cum = cumsum(wo);
    uword cond = lw - 1;
    for(uword j = 0; j < lp ; j++){
        double k = 0.25 * (j + 1);
        for(uword i = 0; i < lw; i++){
            if(Cum(i) > k){
                cond = i - 1;
                break;
            }
        }
        if(cond < 0)
        {
            cond = 0;
        }
        q.row(j) = xo[cond];
        cond = lw - 1;
    }
    return q;
}

bool GWAverage::isValid()
{
    if (SpatialMonoscaleAlgorithm::isValid())
    {
        if (!(mX.n_cols > 0))
            return false;

        return true;
    }
    else return false;
}

void GWAverage::run()
{
    GWM_LOG_STAGE("Initializing");
    uword nRp = mCoords.n_rows, nVar = mX.n_cols;
    createDistanceParameter();
    GWM_LOG_STOP_RETURN(mStatus, void());

    mLocalMean = mat(nRp, nVar, fill::zeros);
    mStandardDev = mat(nRp, nVar, fill::zeros);
    mLocalSkewness = mat(nRp, nVar, fill::zeros);
    mLCV = mat(nRp, nVar, fill::zeros);
    mLVar = mat(nRp, nVar, fill::zeros);
    if (mQuantile)
    {
        mLocalMedian = mat(nRp, nVar, fill::zeros);
        mIQR = mat(nRp, nVar, fill::zeros);
        mQI = mat(nRp, nVar, fill::zeros);
    }
    GWM_LOG_STAGE("Calculating");
    (this->*mSummaryFunction)();
    GWM_LOG_STAGE("Finished");
}

void GWAverage::GWAverageSerial()
{
    mat rankX = mX;
    rankX.each_col([&](vec &x) { x = rank(x); });
    uword nVar = mX.n_cols, nRp = mCoords.n_rows;
    for (uword i = 0; i < nRp; i++)
    {
        GWM_LOG_STOP_BREAK(mStatus);
        vec w = mSpatialWeight.weightVector(i);
        double sumw = sum(w);
        vec Wi = w / sumw;
        mLocalMean.row(i) = trans(Wi) * mX;
        if (mQuantile)
        {
            mat quant = mat(3, nVar);
            for (uword j = 0; j < nVar; j++)
            {
                quant.col(j) = findq(mX.col(j), Wi);
            }
            mLocalMedian.row(i) = quant.row(1);
            mIQR.row(i) = quant.row(2) - quant.row(0);
            mQI.row(i) = (2 * quant.row(1) - quant.row(2) - quant.row(0)) / mIQR.row(i);
        }
        mat centerized = mX.each_row() - mLocalMean.row(i);
        mLVar.row(i) = Wi.t() * (centerized % centerized);
        mStandardDev.row(i) = sqrt(mLVar.row(i));
        mLocalSkewness.row(i) = (Wi.t() * (centerized % centerized % centerized)) / (mLVar.row(i) % mStandardDev.row(i));
        GWM_LOG_PROGRESS(i + 1, nRp);
    }
    mLCV = mStandardDev / mLocalMean;
}

#ifdef ENABLE_OPENMP
void GWAverage::GWAverageOmp()
{
    mat rankX = mX;
    rankX.each_col([&](vec &x) { x = rank(x); });
    uword nVar = mX.n_cols;
    uword nRp = mCoords.n_rows;
#pragma omp parallel for num_threads(mOmpThreadNum)
    for (uword i = 0; i < nRp; i++)
    {
        GWM_LOG_STOP_CONTINUE(mStatus);
        vec w = mSpatialWeight.weightVector(i);
        double sumw = sum(w);
        vec Wi = w / sumw;
        mLocalMean.row(i) = trans(Wi) * mX;
        if (mQuantile)
        {
            mat quant = mat(3, nVar);
            for (uword j = 0; j < nVar; j++)
            {
                quant.col(j) = findq(mX.col(j), Wi);
            }
            mLocalMedian.row(i) = quant.row(1);
            mIQR.row(i) = quant.row(2) - quant.row(0);
            mQI.row(i) = (2 * quant.row(1) - quant.row(2) - quant.row(0)) / mIQR.row(i);
        }
        mat centerized = mX.each_row() - mLocalMean.row(i);
        mLVar.row(i) = Wi.t() * (centerized % centerized);
        mStandardDev.row(i) = sqrt(mLVar.row(i));
        mLocalSkewness.row(i) = (Wi.t() * (centerized % centerized % centerized)) / (mLVar.row(i) % mStandardDev.row(i));
        GWM_LOG_PROGRESS(i + 1, nRp);
    }
    mLCV = mStandardDev / mLocalMean;
}
#endif

void GWAverage::createCalibrationDistanceParameter(const arma::mat& locations)
{
    if (mSpatialWeight.distance()->type() == Distance::DistanceType::CRSDistance || 
        mSpatialWeight.distance()->type() == Distance::DistanceType::MinkwoskiDistance)
    {
        mSpatialWeight.distance()->makeParameter({ locations, mCoords });
    }
}

void GWAverage::calibration(const mat& locations, const mat& x)
{
    GWM_LOG_STAGE("Initializing calibration");
    uword nRp = locations.n_rows, nVar = x.n_cols;
    createCalibrationDistanceParameter(locations);
    GWM_LOG_STOP_RETURN(mStatus, void());

    mLocalMean = mat(nRp, nVar, fill::zeros);
    mStandardDev = mat(nRp, nVar, fill::zeros);
    mLocalSkewness = mat(nRp, nVar, fill::zeros);
    mLCV = mat(nRp, nVar, fill::zeros);
    mLVar = mat(nRp, nVar, fill::zeros);
    if (mQuantile)
    {
        mLocalMedian = mat(nRp, nVar, fill::zeros);
        mIQR = mat(nRp, nVar, fill::zeros);
        mQI = mat(nRp, nVar, fill::zeros);
    }
    GWM_LOG_STAGE("Calibration calculating");
    mat rankX = x;
    rankX.each_col([&](vec &x) { x = rank(x); });
    for (uword i = 0; i < nRp; i++)
    {
        GWM_LOG_STOP_BREAK(mStatus);
        vec w = mSpatialWeight.weightVector(i);
        double sumw = sum(w);
        vec Wi = w / sumw;
        mLocalMean.row(i) = trans(Wi) * x;
        if (mQuantile)
        {
            mat quant = mat(3, nVar);
            for (uword j = 0; j < nVar; j++)
            {
                quant.col(j) = findq(x.col(j), Wi);
            }
            mLocalMedian.row(i) = quant.row(1);
            mIQR.row(i) = quant.row(2) - quant.row(0);
            mQI.row(i) = (2 * quant.row(1) - quant.row(2) - quant.row(0)) / mIQR.row(i);
        }
        mat centerized = x.each_row() - mLocalMean.row(i);
        mLVar.row(i) = Wi.t() * (centerized % centerized);
        mStandardDev.row(i) = sqrt(mLVar.row(i));
        mLocalSkewness.row(i) = (Wi.t() * (centerized % centerized % centerized)) / (mLVar.row(i) % mStandardDev.row(i));
        GWM_LOG_PROGRESS(i + 1, nRp);
    }
    mLCV = mStandardDev / mLocalMean;
}

void GWAverage::setParallelType(const ParallelType &type)
{
    if (type & parallelAbility())
    {
        mParallelType = type;
        updateCalculator();
    }
}

void GWAverage::updateCalculator()
{
    switch (mParallelType)
    {
    case ParallelType::SerialOnly:
        mSummaryFunction = &GWAverage::GWAverageSerial;
        break;
#ifdef ENABLE_OPENMP
    case ParallelType::OpenMP:
        mSummaryFunction = &GWAverage::GWAverageOmp;
        break;
#endif
    default:
        mSummaryFunction = &GWAverage::GWAverageSerial;
        break;
    }
}
