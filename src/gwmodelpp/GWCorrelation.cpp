#include "GWCorrelation.h"
#include "BandwidthSelector.h"
#include "GWRBase.h"
#include <assert.h>
#include "Logger.h"
#include <map>

#ifdef ENABLE_OPENMP
#include <omp.h>
#endif

using namespace std;
using namespace arma;
using namespace gwm;

unordered_map<GWCorrelation::BandwidthInitilizeType,string> GWCorrelation::BandwidthInitilizeTypeNameMapper = {
    make_pair(GWCorrelation::BandwidthInitilizeType::Null, ("Not initilized, not specified")),
    make_pair(GWCorrelation::BandwidthInitilizeType::Initial, ("Initilized")),
    make_pair(GWCorrelation::BandwidthInitilizeType::Specified, ("Specified"))
};

unordered_map<GWCorrelation::BandwidthSelectionCriterionType,string> GWCorrelation::BandwidthSelectionCriterionTypeNameMapper = {
    make_pair(GWCorrelation::BandwidthSelectionCriterionType::CV, ("CV")),
    make_pair(GWCorrelation::BandwidthSelectionCriterionType::AIC, ("AIC"))
};

vec GWCorrelation::del(vec x, uword rowcount){
    vec res;
    if (rowcount == 0)
        res = x.rows(rowcount + 1, x.n_rows - 1);
    else if (rowcount == x.n_rows - 1)
        res = x.rows(0,x.n_rows-2);
    else
        res = join_cols(x.rows(0,rowcount - 1),x.rows(rowcount + 1, x.n_rows - 1));
    return res;
}

void GWCorrelation::run()
{
    GWM_LOG_STAGE("Initializing");
    uword nDp = mCoords.n_rows, nVar = mX.n_cols, nRsp=mY.n_cols;
    uword nCol = nVar * nRsp;
    createDistanceParameter(nCol);
    GWM_LOG_STOP_RETURN(mStatus, void());

    mLVar = mat(nDp, nVar, fill::zeros);
    mLocalMean = mat(nDp, nVar, fill::zeros);
    mCovmat = mat(nDp, nCol, fill::zeros);
    mCorrmat = mat(nDp, nCol, fill::zeros);
    mSCorrmat = mat(nDp, nCol, fill::zeros);

    GWM_LOG_STAGE("Initializing bandwidths");
    for (uword i = 0; i < nCol; i++)
    {
        if(mBandwidthInitilize[i] == BandwidthInitilizeType::Null)
        {
            GWM_LOG_STAGE("Bandwidth optimization")
            mBandwidthSizeCriterion = bandwidthSizeCriterionVar(mBandwidthSelectionApproach[i]);
            mBandwidthSelectionCurrentIndex = i;
            mXi = mX.col(i/nRsp);
            mYi = mY.col((i+nRsp)%nRsp);
            GWM_LOG_INFO(string(GWM_LOG_TAG_GWCORR_INITIAL_BW) + to_string(i));
            BandwidthWeight* bw0 = bandwidth(i);
            BandwidthSelector selector;
            selector.setBandwidth(bw0);
            selector.setLower(bw0->adaptive() ? 20 : 0.0);
            selector.setUpper(bw0->adaptive() ? nDp : mSpatialWeights[i].distance()->maxDistance());
            BandwidthWeight* bw = selector.optimize(this);
            if(bw)
            {
                mSpatialWeights[i].setWeight(bw);
            }
            GWM_LOG_INFO(string(GWM_LOG_TAG_GWCORR_INITIAL_BW)  + to_string(i)  + "," + to_string(bw->bandwidth()));
        }
        GWM_LOG_STOP_RETURN(mStatus, void());
    }

    GWM_LOG_STAGE("Calculating");
    (this->*mSummaryFunction)();
}

void GWCorrelation::GWCorrelationSerial()
{
    mat rankX = mX;
    rankX.each_col([&](vec &x) { x = rank(x); });
    mat rankY = mY;
    rankY.each_col([&](vec y) { y = rank(y); });
    uword nRp = mCoords.n_rows, nVar = mX.n_cols, nRsp=mY.n_cols;
    uword nCol = nVar * nRsp;
    for (uword col = 0; col < nCol; col++)
    {
        for (uword i = 0; i < nRp; i++)
        {
            GWM_LOG_STOP_BREAK(mStatus);
            vec w = mSpatialWeights[col].weightVector(i);
            double sumw = sum(w);
            vec Wi = w / sumw;
            mLocalMean.row(i) = trans(Wi) * mX;
            mat centerized = mX.each_row() - mLocalMean.row(i);
            mLVar.row(i) = Wi.t() * (centerized % centerized);
            uword coly = col / nVar;
            uword colx = (col + nVar) % nVar;
            //correlation
            double covjk = covwt(mY.col(coly), mX.col(colx), Wi);
            double sumW2 = sum(Wi % Wi);
            double covjj = mLVar(i, colx) / (1.0 - sumW2);
            double covkk = mLVar(i, coly+nVar) / (1.0 - sumW2);
            mCovmat(i, col) = covjk;
            mCorrmat(i, col) = covjk / sqrt(covjj * covkk);
            mSCorrmat(i, col) = corwt(rankY.col(coly), rankX.col(colx), Wi);
            GWM_LOG_PROGRESS(i + 1, nRp);
        }
    }
}

void GWCorrelation::setBandwidthSelectionApproach(const vector<BandwidthSelectionCriterionType> &bandwidthSelectionApproach)
{
    if (bandwidthSelectionApproach.size() == (mX.n_cols*mY.n_cols))
    {
        mBandwidthSelectionApproach = bandwidthSelectionApproach;
    }
    else
    {
        length_error e("bandwidthSelectionApproach size do not match input");
        GWM_LOG_ERROR(e.what());
        throw e;
    }  
}

void GWCorrelation::setBandwidthInitilize(const vector<BandwidthInitilizeType> &bandwidthInitilize)
{
    if(bandwidthInitilize.size() == (mX.n_cols*mY.n_cols)){
        mBandwidthInitilize = bandwidthInitilize;
    }
    else
    {
        length_error e("BandwidthInitilize size do not match input");
        GWM_LOG_ERROR(e.what());
        throw e;
    }   
}


void GWCorrelation::setParallelType(const ParallelType& type)
{
    if (type & parallelAbility())
    {
        mParallelType = type;
        switch (type) {
        case ParallelType::SerialOnly:
            mSummaryFunction = &GWCorrelation::GWCorrelationSerial;
            break;
#ifdef ENABLE_OPENMP
        case ParallelType::OpenMP:
            mSummaryFunction = &GWCorrelation::GWCorrelationOmp;
            break;
#endif
        default:
            mSummaryFunction = &GWCorrelation::GWCorrelationSerial;
            break;
        }
        // updateCalculator();
    }
}


#ifdef ENABLE_OPENMP
void GWCorrelation::GWCorrelationOmp()
{
    mat rankX = mX;
    rankX.each_col([&](vec &x) { x = rank(x); });
    mat rankY = mY;
    rankY.each_col([&](vec y) { y = rank(y); });
    uword nRp = mCoords.n_rows, nVar = mX.n_cols, nRsp=mY.n_cols;
    uword nCol = nVar * nRsp;
#pragma omp parallel for num_threads(mOmpThreadNum)
    for (uword col = 0; col < nCol; col++)
    {
        for (uword i = 0; i < nRp; i++)
        {
            GWM_LOG_STOP_BREAK(mStatus);
            vec w = mSpatialWeights[col].weightVector(i);
            double sumw = sum(w);
            vec Wi = w / sumw;
            mLocalMean.row(i) = trans(Wi) * mX;
            mat centerized = mX.each_row() - mLocalMean.row(i);
            mLVar.row(i) = Wi.t() * (centerized % centerized);
            uword coly = col / nVar;
            uword colx = (col + nVar) % nVar;
            double covjk = covwt(mY.col(coly), mX.col(colx), Wi);
            double sumW2 = sum(Wi % Wi);
            double covjj = mLVar(i, colx) / (1.0 - sumW2);
            double covkk = mLVar(i, coly+nVar) / (1.0 - sumW2);
            mCovmat(i, col) = covjk;
            mCorrmat(i, col) = covjk / sqrt(covjj * covkk);
            mSCorrmat(i, col) = corwt(rankY.col(coly), rankX.col(colx), Wi);
            GWM_LOG_PROGRESS(i + 1, nRp);
        }
    }
}
#endif

double GWCorrelation::bandwidthSizeCriterionAICSerial(BandwidthWeight *bandwidthWeight)
{
    int var = mBandwidthSelectionCurrentIndex;
    uword nDp = mCoords.n_rows;
    mat betas(1, nDp, fill::zeros);
    vec shat(2, fill::zeros);
    for (uword i = 0; i < nDp; i++)
    {
        vec d = mSpatialWeights[var].distance()->distance(i);
        vec w = bandwidthWeight->weight(d);
        mat xtw = trans(mXi % w);
        mat xtwx = xtw * mXi;
        mat xtwy = xtw * mYi;
        try
        {
            mat xtwx_inv = inv_sympd(xtwx);
            betas.col(i) = xtwx_inv * xtwy;
            mat ci = xtwx_inv * xtw;
            mat si = mXi(i) * ci;
            shat(0) += si(0, i);
            shat(1) += det(si * si.t());
        }
        catch (const exception& e)
        {
            return DBL_MAX;
        }
    }
    double value = GWRBase::AICc(mX, mY, betas.t(), shat);
    if (mStatus == Status::Success && isfinite(value))
    {
        GWM_LOG_PROGRESS_PERCENT(exp(- abs(mBandwidthLastCriterion - value)));
        mBandwidthLastCriterion = value;
        return value;
    }
    else return DBL_MAX;

    return DBL_MAX;
}


double GWCorrelation::bandwidthSizeCriterionCVSerial(BandwidthWeight *bandwidthWeight)
{
    int var = mBandwidthSelectionCurrentIndex;
    uword nDp = mCoords.n_rows;
    vec shat(2, fill::zeros);
    double cv = 0.0;
    for (uword i = 0; i < nDp; i++)
    {
        vec d = mSpatialWeights[var].distance()->distance(i);
        vec w = bandwidthWeight->weight(d);
        w(i) = 0.0;
        mat xtw = trans(mXi % w);
        mat xtwx = xtw * mXi;
        mat xtwy = xtw * mYi;
        try
        {
            mat xtwx_inv = inv_sympd(xtwx);
            vec beta = xtwx_inv * xtwy;
            double res = mYi(i) - det(mXi.row(i) * beta);
            cv += res * res;
        }
        catch (const exception& e)
        {
            return DBL_MAX;
        }
    }
    if (mStatus == Status::Success && isfinite(cv))
    {
        GWM_LOG_PROGRESS_PERCENT(exp(- abs(mBandwidthLastCriterion - cv)));
        mBandwidthLastCriterion = cv;
        return cv;
    }
    else return DBL_MAX;

    return DBL_MAX;
}

#ifdef ENABLE_OPENMP
double GWCorrelation::bandwidthSizeCriterionAICOmp(BandwidthWeight *bandwidthWeight)
{
    int var = mBandwidthSelectionCurrentIndex;
    uword nDp = mCoords.n_rows;
    mat betas(1, nDp, fill::zeros);
    mat shat_all(2, mOmpThreadNum, fill::zeros);
    bool flag = true;
#pragma omp parallel for num_threads(mOmpThreadNum)
    for (uword i = 0; i < nDp; i++)
    {
        GWM_LOG_STOP_CONTINUE(mStatus);
        if (flag)
        {
            int thread = omp_get_thread_num();
            vec d = mSpatialWeights[var].distance()->distance(i);
            vec w = bandwidthWeight->weight(d);
            mat xtw = trans(mXi % w);
            mat xtwx = xtw * mXi;
            mat xtwy = xtw * mYi;
            try
            {
                mat xtwx_inv = inv_sympd(xtwx);
                betas.col(i) = xtwx_inv * xtwy;
                mat ci = xtwx_inv * xtw;
                mat si = mXi.row(i) * ci;
                shat_all(0, thread) += si(0, i);
                shat_all(1, thread) += det(si * si.t());
            }
            catch (const exception &e)
            {
                GWM_LOG_ERROR(e.what());
                flag = false;
            }
        }
    }

    if (mStatus == Status::Success && flag)
    {
        vec shat = sum(shat_all, 1);
        double value = GWRBase::AICc(mX, mY, betas.t(), shat);
        if (isfinite(value))
        {
            GWM_LOG_PROGRESS_PERCENT(exp(-abs(mBandwidthLastCriterion - value)));
            mBandwidthLastCriterion = value;
            return value;
        }
        else
            return DBL_MAX;
    }
    else
        return DBL_MAX;

    return DBL_MAX;
}
#endif

#ifdef ENABLE_OPENMP
double GWCorrelation::bandwidthSizeCriterionCVOmp(BandwidthWeight *bandwidthWeight)
{
    int var = mBandwidthSelectionCurrentIndex;
    uword nDp = mCoords.n_rows;
    vec shat(2, fill::zeros);
    vec cv_all(mOmpThreadNum, fill::zeros);
    bool flag = true;
#pragma omp parallel for num_threads(mOmpThreadNum)
    for (uword i = 0; i < nDp; i++)
    {
        GWM_LOG_STOP_CONTINUE(mStatus);
        if (flag)
        {
            int thread = omp_get_thread_num();
            vec d = mSpatialWeights[var].distance()->distance(i);
            vec w = bandwidthWeight->weight(d);
            w(i) = 0.0;
            mat xtw = trans(mXi % w);
            mat xtwx = xtw * mXi;
            mat xtwy = xtw * mYi;
            try
            {
                mat xtwx_inv = inv_sympd(xtwx);
                vec beta = xtwx_inv * xtwy;
                double res = mYi(i) - det(mXi(i) * beta);
                if (isfinite(res))
                    cv_all(thread) += res * res;
                else
                    flag = false;
            }
            catch (const exception &e)
            {
                GWM_LOG_ERROR(e.what());
                flag = false;
            }
        }
    }
    if (mStatus == Status::Success && flag)
    {
        double cv = sum(cv_all);
        GWM_LOG_PROGRESS_PERCENT(exp(-abs(mBandwidthLastCriterion - cv)));
        mBandwidthLastCriterion = cv;
        return cv;
    }
    else
        return DBL_MAX;

    return DBL_MAX;
}
#endif