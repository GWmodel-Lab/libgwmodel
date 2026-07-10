#include "GWAverage.h"
#include <assert.h>
#include <algorithm>
#include <map>
#include "BandwidthSelector.h"
#include "Logger.h"
#include <limits>

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
    if (x.n_rows != w.n_rows || x.n_rows == 0 || !std::all_of(w.begin(), w.end(), [](double v) { return std::isfinite(v); }))
    {
        return vec(3, fill::zeros);
    }

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

    if (mIsAutoselectBandwidth)
    {
        GWM_LOG_STAGE("Bandwidth selection");
        BandwidthWeight* bw0 = mSpatialWeight.weight<BandwidthWeight>();
        double lower = bw0->adaptive() ? 2.0 : mSpatialWeight.distance()->minDistance();
        double upper = bw0->adaptive() ? (double)nRp : mSpatialWeight.distance()->maxDistance();

        GWM_LOG_INFO(IBandwidthSelectable::infoBandwidthCriterion(bw0));
        BandwidthSelector selector(bw0, lower, upper);
        BandwidthWeight* bw = selector.optimize(this);
        if (bw)
        {
            mSpatialWeight.setWeight(bw);
#ifdef ENABLE_CUDA
            if (mParallelType & ParallelType::CUDA)
            {
                mSpatialWeight.prepareCuda(mGpuId);
            }
#endif // ENABLE_CUDA
            mBandwidthSelectionCriterionList = selector.bandwidthCriterion();
        }
    }

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
        if (w.n_rows != nRp || !std::all_of(w.begin(), w.end(), [](double v) { return std::isfinite(v); }))
        {
            w = vec(nRp, fill::zeros);
        }
        double sumw = sum(w);
        vec Wi = arma::zeros<vec>(nRp);
        if (isfinite(sumw) && sumw != 0.0)
        {
            Wi = w / sumw;
        }
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
            rowvec qi = (2 * quant.row(1) - quant.row(2) - quant.row(0));
            for (uword j = 0; j < nVar; j++)
            {
                double denom = mIQR(i, j);
                mQI(i, j) = (isfinite(denom) && denom != 0.0) ? qi(j) / denom : 0.0;
            }
        }
        mat centerized = mX.each_row() - mLocalMean.row(i);
        mLVar.row(i) = Wi.t() * (centerized % centerized);
        mStandardDev.row(i) = sqrt(mLVar.row(i));
        rowvec denom = mLVar.row(i) % mStandardDev.row(i);
        rowvec numerator = trans(Wi) * (centerized % centerized % centerized);
        mLocalSkewness.row(i).zeros();
        for (uword j = 0; j < nVar; j++)
        {
            if (isfinite(denom(j)) && denom(j) != 0.0)
            {
                mLocalSkewness(i, j) = numerator(j) / denom(j);
            }
            else
            {
                mLocalSkewness(i, j) = 0.0;
            }
        }
        GWM_LOG_PROGRESS(i + 1, nRp);
    }
    mLCV.zeros();
    for (uword i = 0; i < nRp; i++)
    {
        for (uword j = 0; j < nVar; j++)
        {
            double meanVal = mLocalMean(i, j);
            if (isfinite(meanVal) && meanVal != 0.0)
            {
                mLCV(i, j) = mStandardDev(i, j) / meanVal;
            }
            else
            {
                mLCV(i, j) = 0.0;
            }
        }
    }
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
        if (w.n_rows != nRp || !std::all_of(w.begin(), w.end(), [](double v) { return std::isfinite(v); }))
        {
            w = vec(nRp, fill::zeros);
        }
        double sumw = sum(w);
        vec Wi = arma::zeros<vec>(nRp);
        if (isfinite(sumw) && sumw != 0.0)
        {
            Wi = w / sumw;
        }
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
            rowvec qi = (2 * quant.row(1) - quant.row(2) - quant.row(0));
            for (uword j = 0; j < nVar; j++)
            {
                double denom = mIQR(i, j);
                mQI(i, j) = (isfinite(denom) && denom != 0.0) ? qi(j) / denom : 0.0;
            }
        }
        mat centerized = mX.each_row() - mLocalMean.row(i);
        mLVar.row(i) = Wi.t() * (centerized % centerized);
        mStandardDev.row(i) = sqrt(mLVar.row(i));
        rowvec denom = mLVar.row(i) % mStandardDev.row(i);
        rowvec numerator = trans(Wi) * (centerized % centerized % centerized);
        mLocalSkewness.row(i).zeros();
        for (uword j = 0; j < nVar; j++)
        {
            if (isfinite(denom(j)) && denom(j) != 0.0)
            {
                mLocalSkewness(i, j) = numerator(j) / denom(j);
            }
            else
            {
                mLocalSkewness(i, j) = 0.0;
            }
        }
        GWM_LOG_PROGRESS(i + 1, nRp);
    }
    mLCV.zeros();
    for (uword i = 0; i < nRp; i++)
    {
        for (uword j = 0; j < nVar; j++)
        {
            double meanVal = mLocalMean(i, j);
            if (isfinite(meanVal) && meanVal != 0.0)
            {
                mLCV(i, j) = mStandardDev(i, j) / meanVal;
            }
            else
            {
                mLCV(i, j) = 0.0;
            }
        }
    }
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
        vec Wi = arma::zeros<vec>(w.n_rows);
        if (isfinite(sumw) && sumw != 0.0)
        {
            Wi = w / sumw;
        }
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
            rowvec qi = (2 * quant.row(1) - quant.row(2) - quant.row(0));
            for (uword j = 0; j < nVar; j++)
            {
                double denom = mIQR(i, j);
                mQI(i, j) = (isfinite(denom) && denom != 0.0) ? qi(j) / denom : 0.0;
            }
        }
        mat centerized = x.each_row() - mLocalMean.row(i);
        mLVar.row(i) = Wi.t() * (centerized % centerized);
        mStandardDev.row(i) = sqrt(mLVar.row(i));
        rowvec denom = mLVar.row(i) % mStandardDev.row(i);
        rowvec numerator = trans(Wi) * (centerized % centerized % centerized);
        mLocalSkewness.row(i).zeros();
        for (uword j = 0; j < nVar; j++)
        {
            if (isfinite(denom(j)) && denom(j) != 0.0)
            {
                mLocalSkewness(i, j) = numerator(j) / denom(j);
            }
            else
            {
                mLocalSkewness(i, j) = 0.0;
            }
        }
        GWM_LOG_PROGRESS(i + 1, nRp);
    }
    mLCV.zeros();
    for (uword i = 0; i < nRp; i++)
    {
        for (uword j = 0; j < nVar; j++)
        {
            double meanVal = mLocalMean(i, j);
            if (isfinite(meanVal) && meanVal != 0.0)
            {
                mLCV(i, j) = mStandardDev(i, j) / meanVal;
            }
            else
            {
                mLCV(i, j) = 0.0;
            }
        }
    }
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

Status GWAverage::getCriterion(BandwidthWeight* weight, double& criterion)
{
    BandwidthWeight* currentBw = mSpatialWeight.weight<BandwidthWeight>();
    if (!currentBw || !weight)
    {
        criterion = DBL_MAX;
        return Status::Success;
    }

    BandwidthWeight* backup = static_cast<BandwidthWeight*>(currentBw->clone());
    mSpatialWeight.setWeight(weight);

    uword n = mCoords.n_rows;
    uword p = mX.n_cols;
    if (n < 2 || p == 0)
    {
        criterion = DBL_MAX;
        mSpatialWeight.setWeight(backup);
        delete backup;
        return Status::Success;
    }

    double totalF = 0.0;
    bool valid = false;

    for (uword j = 0; j < p; j++)
    {
        mat weightedVals(n, n, fill::zeros);
        for (uword i = 0; i < n; i++)
        {
            vec w = mSpatialWeight.weightVector(i);
            if (w.n_rows != n)
                continue;
            weightedVals.col(i) = w % mX.col(j);
        }

        double yMean = accu(weightedVals) / double(n * n);
        mat centered = weightedVals - yMean;
        double SST = accu(centered % centered);
        if (!isfinite(SST) || SST < 0.0)
            continue;

        rowvec groupMeans = mean(weightedVals, 0);
        double SSE = 0.0;
        for (uword i = 0; i < n; i++)
        {
            vec diff = weightedVals.col(i) - groupMeans(i);
            SSE += accu(diff % diff);
        }

        double SSB = 0.0;
        for (uword i = 0; i < n; i++)
        {
            double meanDiff = groupMeans(i) - yMean;
            SSB += double(n) * meanDiff * meanDiff;
        }
        double MSB = SSB / double(n - 1);
        double MSW = SSE / double(n * n - n);
        if (!isfinite(MSB) || !isfinite(MSW) || MSW == 0.0)
            continue;

        double F = MSB / MSW;
        if (!isfinite(F))
            continue;

        totalF += F;
        valid = true;
    }

    criterion = valid ? -totalF : DBL_MAX;

    mSpatialWeight.setWeight(backup);
    delete backup;
    return Status::Success;
}
