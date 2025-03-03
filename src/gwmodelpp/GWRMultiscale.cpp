#include "GWRMultiscale.h"
#ifdef ENABLE_OPENMP
#include <omp.h>
#endif
#include <exception>
#include <vector>
#include <string>
#include <spatialweight/CRSDistance.h>
#include "BandwidthSelector.h"
#include "VariableForwardSelector.h"
#include "Logger.h"
#include "GWRBasic.h"

#ifdef ENABLE_CUDA
#include <cuda_runtime.h>
#include "cumat.hpp"
#include "CudaUtils.h"
#endif // ENABLE_CUDA

#ifdef ENABLE_MPI
#include <mpi.h>
#include "gwmodelpp/utils/armampi.h"
#endif

using namespace std;
using namespace arma;
using namespace gwm;

int GWRMultiscale::treeChildCount = 0;

enum class GWRMultiscaleFitMPITags
{
    Betas = 1 << 0,
    SMat
};

unordered_map<GWRMultiscale::BandwidthInitilizeType,string> GWRMultiscale::BandwidthInitilizeTypeNameMapper = {
    make_pair(GWRMultiscale::BandwidthInitilizeType::Null, ("Not initilized, not specified")),
    make_pair(GWRMultiscale::BandwidthInitilizeType::Initial, ("Initilized")),
    make_pair(GWRMultiscale::BandwidthInitilizeType::Specified, ("Specified"))
};

unordered_map<GWRMultiscale::BandwidthSelectionCriterionType,string> GWRMultiscale::BandwidthSelectionCriterionTypeNameMapper = {
    make_pair(GWRMultiscale::BandwidthSelectionCriterionType::CV, ("CV")),
    make_pair(GWRMultiscale::BandwidthSelectionCriterionType::AIC, ("AIC"))
};

unordered_map<GWRMultiscale::BackFittingCriterionType,string> GWRMultiscale::BackFittingCriterionTypeNameMapper = {
    make_pair(GWRMultiscale::BackFittingCriterionType::CVR, ("CVR")),
    make_pair(GWRMultiscale::BackFittingCriterionType::dCVR, ("dCVR"))
};

RegressionDiagnostic GWRMultiscale::CalcDiagnostic(const mat &x, const vec &y, const vec &shat, double RSS)
{
    // 诊断信息
    double nDp = (double)x.n_rows;
    double RSSg = RSS;
    double sigmaHat21 = RSSg / nDp;
    double TSS = sum((y - mean(y)) % (y - mean(y)));
    double Rsquare = 1 - RSSg / TSS;

    double trS = shat(0);
    double trStS = shat(1);
    double enp = 2 * trS - trStS;
    double edf = nDp - 2 * trS + trStS;
    double AICc = nDp * log(sigmaHat21) + nDp * log(2 * M_PI) + nDp * ((nDp + trS) / (nDp - 2 - trS));
    double adjustRsquare = 1 - (1 - Rsquare) * (nDp - 1) / (edf - 1);

    // 保存结果
    RegressionDiagnostic diagnostic;
    diagnostic.RSS = RSSg;
    diagnostic.AICc = AICc;
    diagnostic.ENP = enp;
    diagnostic.EDF = edf;
    diagnostic.RSquareAdjust = adjustRsquare;
    diagnostic.RSquare = Rsquare;
    return diagnostic;
}

mat GWRMultiscale::fit()
{
    uword nDp = mX.n_rows, nVar = mX.n_cols;
    mWorkRange = make_pair(uword(0), nDp);

    // ********************************
    // Centering and scaling predictors
    // ********************************
    GWM_LOG_STAGE("Centering and scaling predictors");
    mX0 = mX;
    mY0 = mY;
    for (uword i = (mHasIntercept ? 1 : 0); i < nVar ; i++)
    {
        if (mPreditorCentered[i])
        {
            mX.col(i) = mX.col(i) - mean(mX.col(i));
        }
    }
    GWM_LOG_STOP_RETURN(mStatus, mat(nDp, nVar, arma::fill::zeros));

    // *****************************************************
    // Calculate the initial beta0 from the above bandwidths
    // *****************************************************
    GWM_LOG_STAGE("Calculating initial betas");
    mat betas = fitInitial();
    GWM_LOG_STOP_RETURN(mStatus, mat(nDp, nVar, arma::fill::zeros));

    GWM_LOG_STAGE("Initializing");
    createDistanceParameter(nVar);
    mMaxDistances.resize(nVar);
    mMinDistances.resize(nVar);
    
#ifdef ENABLE_MPI
    if (mParallelType & ParallelType::MPI)
    {
        uword aShape[4];
        if (mWorkerId == 0) // master process
        {
            // sync matrix dimensions
            uword nDim = mCoords.n_cols;
            mWorkRangeSize = nDp / uword(mWorkerNum) + ((nDp % uword(mWorkerNum) == 0) ? 0 : 1);
            aShape[0] = nDp;
            aShape[1] = nVar;
            aShape[2] = nDim;
            aShape[3] = mWorkRangeSize;
        }
        MPI_Bcast(aShape, 4, GWM_MPI_UWORD, 0, MPI_COMM_WORLD);
        if (mWorkerId > 0)                // slave process
        {
            nDp = aShape[0];
            nVar = aShape[1];
            mX.resize(nDp, nVar);
            mY.resize(nDp);
            mCoords.resize(aShape[0], aShape[2]);
            mWorkRangeSize = aShape[3];
        }
        MPI_Bcast(mX.memptr(), mX.n_elem, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(mY.memptr(), mY.n_elem, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(mCoords.memptr(), mCoords.n_elem, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        uword workRangeFrom = uword(mWorkerId) * mWorkRangeSize;
        mWorkRange = make_pair(workRangeFrom, min(workRangeFrom + mWorkRangeSize, nDp));
    }
#endif // ENABLE_MPI
#ifdef ENABLE_CUDA
    if (mParallelType & ParallelType::CUDA)
    {
        cubase::create_handle();
        for (size_t i = 0; i < nVar; i++)
        {
            mSpatialWeights[i].prepareCuda(mGpuId);
        }
    }
#endif // ENABLE_CUDA
    for (size_t i = 0; i < nVar; i++)
    {
        mMinDistances[i] = mSpatialWeights[i].distance()->minDistance();
        mMaxDistances[i] = mSpatialWeights[i].distance()->maxDistance();
    }
    GWM_LOG_STOP_RETURN(mStatus, mat(nDp, nVar, arma::fill::zeros));

    // ***********************
    // Intialize the bandwidth
    // ***********************
    GWM_LOG_STAGE("Calculating initial bandwidths");
    mYi = mY;
    for (uword i = 0; i < nVar ; i++)
    {
        GWM_LOG_STOP_BREAK(mStatus);
        if (mBandwidthInitilize[i] == BandwidthInitilizeType::Null)
        {
            mBandwidthSizeCriterion = bandwidthSizeCriterionVar(mBandwidthSelectionApproach[i]);
            mBandwidthSelectionCurrentIndex = i;
            mXi = mX.col(i);
            BandwidthWeight* bw0 = bandwidth(i);
            bool adaptive = bw0->adaptive();

            GWM_LOG_INFO(string(GWM_LOG_TAG_MGWR_INITIAL_BW) + to_string(i));
            BandwidthSelector selector;
            selector.setBandwidth(bw0);
            selector.setLower(mGoldenLowerBounds.value_or(adaptive ? mAdaptiveLower : mMinDistances[i]));
            selector.setUpper(mGoldenUpperBounds.value_or(adaptive ? mCoords.n_rows : mMaxDistances[i]));
            BandwidthWeight* bw = selector.optimize(this);
            if (bw)
            {
                mSpatialWeights[i].setWeight(bw);
#ifdef ENABLE_CUDA
                if (mParallelType & ParallelType::CUDA)
                    mSpatialWeights[i].prepareCuda(mGpuId);
#endif // ENABLE_CUDA
            }
            GWM_LOG_INFO(string(GWM_LOG_TAG_MGWR_INITIAL_BW)  + to_string(i)  + "," + to_string(bw->bandwidth()));
        }
        GWM_LOG_STOP_RETURN(mStatus, mat(nDp, nVar, arma::fill::zeros));
    }

    GWM_LOG_STAGE("Initializing diagnostic matrices");
    mat idm = eye(nVar, nVar);
    if (mHasHatMatrix)
    {
        mSArray = cube(mS0.n_rows, mS0.n_cols, nVar, fill::zeros);
        for (uword i = 0; i < nVar; ++i)
        {
            for (uword j = mWorkRange.first; j < mWorkRange.second ; ++j)
            {
                uword e = j - mWorkRange.first;
                mSArray.slice(i).row(e) = mX(j, i) * (idm.row(i) * mC.slice(e));
            }
        }
    }

    GWM_LOG_STAGE("Model fitting");
    mBetas = backfitting(betas);
    GWM_LOG_STOP_RETURN(mStatus, mat(nDp, nVar, arma::fill::zeros));

    // Diagnostic
    GWM_LOG_STAGE("Model Diagnostic");
    vec shat(2, arma::fill::zeros);
    if (mHasHatMatrix)
    {
#ifdef ENABLE_MPI
        if (mParallelType & ParallelType::MPI)
        {
            vec shati(2);
            shati(0) = trace(mS0.cols(mWorkRange.first, mWorkRange.second - 1));
            shati(1) = trace(mS0 * mS0.t());
            MPI_Reduce(shati.memptr(), shat.memptr(), 2, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
            MPI_Bcast(shat.memptr(), 2, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        }
        else
        {
#endif // ENABLE_MPI
            shat = {
                mHasHatMatrix ? trace(mS0) : 0,
                mHasHatMatrix ? trace(mS0 * mS0.t()) : 0
            };
#ifdef ENABLE_MPI
        }
#endif // ENABLE_MPI
        mBetasTV = mBetas / mBetasSE;
    }
    mDiagnostic = CalcDiagnostic(mX, mY, shat, mRSS0);
    vec yhat = Fitted(mX, mBetas);
    vec residual = mY - yhat;

    // Cleaning
#ifdef ENABLE_CUDA
    if (mParallelType & ParallelType::CUDA)
    {
        cubase::destory_handle();
    }
#endif // ENABLE_CUDA

    return mBetas;
}

arma::mat gwm::GWRMultiscale::fitInitial()
{
    GWRBasic gwr;
    gwr.setCoords(mCoords);
    gwr.setDependentVariable(mY);
    gwr.setIndependentVariables(mX);
    gwr.setSpatialWeight(mInitSpatialWeight);
    gwr.setIsAutoselectBandwidth(true);
    switch (mBandwidthSelectionApproach[0])
    {
    case GWRMultiscale::BandwidthSelectionCriterionType::CV:
        gwr.setBandwidthSelectionCriterion(GWRBasic::BandwidthSelectionCriterionType::CV);
        break;
    case GWRMultiscale::BandwidthSelectionCriterionType::AIC:
        gwr.setBandwidthSelectionCriterion(GWRBasic::BandwidthSelectionCriterionType::AIC);
    default:
        break;
    }
    gwr.setParallelType(mParallelType);
    switch (mParallelType)
    {
    case ParallelType::OpenMP:
    case ParallelType::MPI_MP:
        gwr.setOmpThreadNum(mOmpThreadNum);
        break;
    case ParallelType::CUDA:
    case ParallelType::MPI_CUDA:
        gwr.setGroupSize(mGroupLength);
        gwr.setGPUId(mGpuId);
        break;
    default:
        break;
    }
    if (mParallelType & ParallelType::MPI)
    {
        gwr.setWorkerId(mWorkerId);
        gwr.setWorkerNum(mWorkerNum);
    }
    gwr.setStoreS(mHasHatMatrix);
    gwr.setStoreC(mHasHatMatrix);
    if (mGoldenLowerBounds.has_value()) gwr.setGoldenLowerBounds(mGoldenLowerBounds.value());
    if (mGoldenUpperBounds.has_value()) gwr.setGoldenUpperBounds(mGoldenUpperBounds.value());
    mat betas = gwr.fit();
    mBetasSE = gwr.betasSE();
    if (mHasHatMatrix) {
        mS0 = gwr.s();
        mC = gwr.c();
    }
    return betas;
}

mat GWRMultiscale::backfitting(const mat& betas0)
{
    uword nDp = mCoords.n_rows, nVar = mX.n_cols;
    mat betas = betas0;
    // cout << "[backfitting] Process " << mWorkerId << " betas\n";
    GWM_LOG_STOP_RETURN(mStatus, mat(nDp, nVar, arma::fill::zeros));

    // ***********************************************************
    // Select the optimum bandwidths for each independent variable
    // ***********************************************************
    GWM_LOG_MGWR_BACKFITTING("Selecting the optimum bandwidths for each independent variable");
    uvec bwChangeNo(nVar, fill::zeros);
    vec resid = mY - Fitted(mX, betas);
    double RSS0 = sum(resid % resid), RSS1 = DBL_MAX;
    double criterion = DBL_MAX;
    for (size_t iteration = 1; iteration <= mMaxIteration && criterion > mCriterionThreshold; iteration++)
    {
        GWM_LOG_STOP_BREAK(mStatus);
        GWM_LOG_MGWR_BACKFITTING("#iteration " + to_string(iteration));
        for (uword i = 0; i < nVar  ; i++)
        {
            GWM_LOG_STOP_BREAK(mStatus);
            mXi = mX.col(i);
            vec fi = betas.col(i) % mX.col(i);
            mYi = resid + fi;
            if (mBandwidthInitilize[i] != BandwidthInitilizeType::Specified)
            {
                GWM_LOG_MGWR_BACKFITTING("#variable-bandwidth-selection " + to_string(i));
                mBandwidthSizeCriterion = bandwidthSizeCriterionVar(mBandwidthSelectionApproach[i]);
                mBandwidthSelectionCurrentIndex = i;
                BandwidthWeight* bwi0 = bandwidth(i);
                bool adaptive = bwi0->adaptive();
                BandwidthSelector selector;
                selector.setBandwidth(bwi0);
                selector.setLower(mGoldenLowerBounds.value_or(adaptive ? mAdaptiveLower : mMinDistances[i]));
                selector.setUpper(mGoldenUpperBounds.value_or(adaptive ? mCoords.n_rows : mMaxDistances[i]));
                BandwidthWeight* bwi = selector.optimize(this);
                double bwi0s = bwi0->bandwidth(), bwi1s = bwi->bandwidth();
                vector<string> vbs_args {
                    to_string(i),
                    to_string(bwi0s),
                    to_string(bwi1s),
                    to_string(abs(bwi1s - bwi0s))
                };
                if (abs(bwi1s - bwi0s) > mBandwidthSelectThreshold[i])
                {
                    bwChangeNo(i) = 0;
                    vbs_args.push_back("false");
                }
                else
                {
                    bwChangeNo(i) += 1;
                    if (bwChangeNo(i) >= mBandwidthSelectRetryTimes)
                    {
                        mBandwidthInitilize[i] = BandwidthInitilizeType::Specified;
                        vbs_args.push_back("true");
                    }
                    else
                    {
                        vbs_args.push_back(to_string(bwChangeNo(i)));
                        vbs_args.push_back(to_string(mBandwidthSelectRetryTimes - bwChangeNo(i)));
                    }
                }
                mSpatialWeights[i].setWeight(bwi);
#ifdef ENABLE_CUDA
                mSpatialWeights[i].prepareCuda(mGpuId);
#endif // ENABLE_CUDA

                GWM_LOG_MGWR_BACKFITTING("#variable-bandwidth-selection " + strjoin(",", vbs_args));
            }
            GWM_LOG_STOP_BREAK(mStatus);

            vec betai = (this->*mFitVar)(i);
            // betas.brief_print("betas");
            // betai.brief_print("betai");
            betas.col(i) = betai;
            resid = mY - Fitted(mX, betas);
        }
        RSS1 = RSS(mX, mY, betas);
        criterion = (mCriterionType == BackFittingCriterionType::CVR) ?
                    abs(RSS1 - RSS0) :
                    sqrt(abs(RSS1 - RSS0) / RSS1);
        GWM_LOG_MGWR_BACKFITTING("#backfitting-criterion " + to_string(criterion));
        GWM_LOG_PROGRESS_PERCENT(exp(- abs(criterion - mCriterionThreshold)));
        RSS0 = RSS1;
    }
    GWM_LOG_STOP_RETURN(mStatus, betas);

    GWM_LOG_MGWR_BACKFITTING("Finished");
    mRSS0 = RSS0;
    return betas;
}

arma::vec gwm::GWRMultiscale::fitVarBase(const size_t var)
{
    mat S;
    vec beta = (this->*mFitVarCore)(mXi, mYi, mSpatialWeights[var], S);
    if (mHasHatMatrix)
    {
        mat SArrayi = mSArray.slice(var) - mS0;
        mSArray.slice(var) = S * SArrayi + S;
        mS0 = mSArray.slice(var) - SArrayi;
    }
    return beta;
}

double gwm::GWRMultiscale::bandwidthSizeCriterionVarCVBase(BandwidthWeight* bandwidthWeight)
{
    SpatialWeight sw(bandwidthWeight, mSpatialWeights[mBandwidthSelectionCurrentIndex].distance());
    try
    {
        vec betas = (this->*mFitVarCoreCV)(mXi, mYi, sw);
        vec res = mYi - mXi % betas;
        double cv = sum(res % res);
        if (mStatus == Status::Success && isfinite(cv))
        {
            GWM_LOG_INFO(IBandwidthSelectable::infoBandwidthCriterion(bandwidthWeight, cv));
            GWM_LOG_PROGRESS_PERCENT(exp(- abs(mBandwidthLastCriterion - cv)));
            mBandwidthLastCriterion = cv;
            return cv;
        }
        else return DBL_MAX;
    }
    catch(const std::exception& e)
    {
        return DBL_MAX;
    }
}

double gwm::GWRMultiscale::bandwidthSizeCriterionVarAICBase(BandwidthWeight* bandwidthWeight)
{
    SpatialWeight sw(bandwidthWeight, mSpatialWeights[mBandwidthSelectionCurrentIndex].distance());
    try
    {
        vec shat;
        mat betas = (this->*mFitVarCoreSHat)(mXi, mYi, sw, shat);
        double value = GWRBase::AICc(mXi, mYi, betas, shat);
        if (mStatus == Status::Success && isfinite(value))
        {
            GWM_LOG_INFO(IBandwidthSelectable::infoBandwidthCriterion(bandwidthWeight, value));
            GWM_LOG_PROGRESS_PERCENT(exp(- abs(mBandwidthLastCriterion - value)));
            mBandwidthLastCriterion = value;
            return value;
        }
        else return DBL_MAX;
    }
    catch(const std::exception& e)
    {
        return DBL_MAX;
    }
}

bool GWRMultiscale::isValid()
{
    if (!(mX.n_cols > 0))
        return false;

    size_t nVar = mX.n_cols;

    if (mSpatialWeights.size() != nVar)
        return false;

    if (mBandwidthInitilize.size() != nVar)
        return false;

    if (mBandwidthSelectionApproach.size() != nVar)
        return false;

    if (mPreditorCentered.size() != nVar)
        return false;

    if (mBandwidthSelectThreshold.size() != nVar)
        return false;

    for (size_t i = 0; i < nVar; i++)
    {
        BandwidthWeight* bw = mSpatialWeights[i].weight<BandwidthWeight>();
        if (mBandwidthInitilize[i] == GWRMultiscale::Specified || mBandwidthInitilize[i] == GWRMultiscale::Initial)
        {
            if (bw->adaptive())
            {
                if (bw->bandwidth() <= 1)
                    return false;
            }
            else
            {
                if (bw->bandwidth() < 0.0)
                    return false;
            }
        }
    }

    return true;
}

vec GWRMultiscale::fitVarCoreSerial(const vec &x, const vec &y, const SpatialWeight& sw, mat &S)
{
    uword nDp = mCoords.n_rows;
    mat betas(1, nDp, fill::zeros);
    bool success = true;
    std::exception except;
    if (mHasHatMatrix)
    {
        mat ci, si;
        uword rangeSize = mWorkRange.second - mWorkRange.first;
        S = mat(rangeSize, nDp, fill::zeros);
        for (uword i = mWorkRange.first; i < mWorkRange.second; i++)
        {
            GWM_LOG_STOP_BREAK(mStatus);
            vec w = sw.weightVector(i);
            mat xtw = trans(x % w);
            mat xtwx = xtw * x;
            mat xtwy = xtw * y;
            try
            {
                mat xtwx_inv = inv_sympd(xtwx);
                betas.col(i) = xtwx_inv * xtwy;
                ci = xtwx_inv * xtw;
                si = x(i) * ci;
                S.row(i - mWorkRange.first) = si;
            }
            catch (const exception& e)
            {
                GWM_LOG_ERROR(e.what());
                except = e;
                success = false;
            }
            GWM_LOG_PROGRESS(i + 1, nDp);
        }
    }
    else
    {
        for (uword i = mWorkRange.first; i < mWorkRange.second; i++)
        {
            GWM_LOG_STOP_BREAK(mStatus);
            vec w = sw.weightVector(i);
            mat xtw = trans(x % w);
            mat xtwx = xtw * x;
            mat xtwy = xtw * y;
            try
            {
                mat xtwx_inv = inv_sympd(xtwx);
                betas.col(i) = xtwx_inv * xtwy;
            }
            catch (const exception& e)
            {
                GWM_LOG_ERROR(e.what());
                except = e;
                success = false;
            }
            GWM_LOG_PROGRESS(i + 1, nDp);
        }
    }
    if (!success)
    {
        throw except;
    }
    return betas.t();
}

vec GWRMultiscale::fitVarCoreCVSerial(const vec &x, const vec &y, const SpatialWeight& sw)
{
    uword nDp = x.n_rows;
    vec beta(nDp, fill::zeros);
    for (uword i = mWorkRange.first; i < mWorkRange.second; i++)
    {
        GWM_LOG_STOP_BREAK(mStatus);
        vec w = sw.weightVector(i);
        w(i) = 0.0;
        mat xtw = trans(x % w);
        mat xtwx = xtw * x;
        mat xtwy = xtw * y;
        try
        {
            mat xtwx_inv = inv_sympd(xtwx);
            beta(i) = as_scalar(xtwx_inv * xtwy);
        }
        catch (const exception& e)
        {
            GWM_LOG_ERROR(e.what());
            throw e;
        }
    }
    return beta;
}

vec GWRMultiscale::fitVarCoreSHatSerial(const vec &x, const vec &y, const SpatialWeight& sw, vec& shat)
{
    uword nDp = x.n_rows;
    vec betas(nDp, fill::zeros);
    shat = vec(2, fill::zeros);
    for (uword i = mWorkRange.first; i < mWorkRange.second; i++)
    {
        GWM_LOG_STOP_BREAK(mStatus);
        vec w = sw.weightVector(i);
        mat xtw = trans(x % w);
        mat xtwx = xtw * x;
        mat xtwy = xtw * y;
        try
        {
            mat xtwx_inv = inv_sympd(xtwx);
            betas(i) = as_scalar(xtwx_inv * xtwy);
            mat ci = xtwx_inv * xtw;
            mat si = x(i) * ci;
            shat(0) += si(0, i);
            shat(1) += as_scalar(si * si.t());
        }
        catch (const exception& e)
        {
            GWM_LOG_ERROR(e.what());
            throw e;
        }
    }
    return betas;
}


#ifdef ENABLE_OPENMP
vec GWRMultiscale::fitVarCoreOmp(const vec &x, const vec &y, const SpatialWeight& sw, mat &S)
{
    uword nDp = mCoords.n_rows;
    mat betas(1, nDp, fill::zeros);
    bool success = true;
    std::exception except;
    if (mHasHatMatrix)
    {
        uword rangeSize = mWorkRange.second - mWorkRange.first;
        S = mat(rangeSize, nDp, fill::zeros);
#pragma omp parallel for num_threads(mOmpThreadNum)
        for (uword i = mWorkRange.first; i < mWorkRange.second; i++)
        {
            GWM_LOG_STOP_CONTINUE(mStatus);
            vec w = sw.weightVector(i);
            mat xtw = trans(x % w);
            mat xtwx = xtw * x;
            mat xtwy = xtw * y;
            try
            {
                mat xtwx_inv = inv_sympd(xtwx);
                betas.col(i) = xtwx_inv * xtwy;
                mat ci = xtwx_inv * xtw;
                mat si = x(i) * ci;
                S.row(i - mWorkRange.first) = si;
            }
            catch (const exception& e)
            {
                GWM_LOG_ERROR(e.what());
                except = e;
                success = false;
            }
            GWM_LOG_PROGRESS(i + 1, nDp);
        }
    }
    else
    {
#pragma omp parallel for num_threads(mOmpThreadNum)
        for (uword i = mWorkRange.first; i < mWorkRange.second; i++)
        {
            GWM_LOG_STOP_CONTINUE(mStatus);
            vec w = sw.weightVector(i);
            mat xtw = trans(x % w);
            mat xtwx = xtw * x;
            mat xtwy = xtw * y;
            try
            {
                mat xtwx_inv = inv_sympd(xtwx);
                betas.col(i) = xtwx_inv * xtwy;
            }
            catch (const exception& e)
            {
                GWM_LOG_ERROR(e.what());
                except = e;
                success = false;
            }
            GWM_LOG_PROGRESS(i + 1, nDp);
        }
    }
    if (!success)
    {
        throw except;
    }
    return betas.t();
}

vec GWRMultiscale::fitVarCoreCVOmp(const vec &x, const vec &y, const SpatialWeight& sw)
{
    uword nDp = mCoords.n_rows;
    vec beta(nDp, fill::zeros);
    bool flag = true;
#pragma omp parallel for num_threads(mOmpThreadNum)
    for (uword i = mWorkRange.first; i < mWorkRange.second; i++)
    {
        GWM_LOG_STOP_CONTINUE(mStatus);
        if (flag)
        {
            vec w = sw.weightVector(i);
            w(i) = 0.0;
            mat xtw = trans(x % w);
            mat xtwx = xtw * x;
            mat xtwy = xtw * y;
            try
            {
                mat xtwx_inv = inv_sympd(xtwx);
                beta(i) = as_scalar(xtwx_inv * xtwy);
            }
            catch (const exception& e)
            {
                GWM_LOG_ERROR(e.what());
                flag = false;
            }
        }
    }
    return beta;
}

vec GWRMultiscale::fitVarCoreSHatOmp(const vec &x, const vec &y, const SpatialWeight& sw, vec& shat)
{
    uword nDp = mCoords.n_rows;
    vec betas(nDp, fill::zeros);
    mat shat_all(2, mOmpThreadNum, fill::zeros);
    bool flag = true;
#pragma omp parallel for num_threads(mOmpThreadNum)
    for (uword i = mWorkRange.first; i < mWorkRange.second; i++)
    {
        GWM_LOG_STOP_CONTINUE(mStatus);
        if (flag)
        {
            int thread = omp_get_thread_num();
            vec w = sw.weightVector(i);
            mat xtw = trans(x % w);
            mat xtwx = xtw * x;
            mat xtwy = xtw * y;
            try
            {
                mat xtwx_inv = inv_sympd(xtwx);
                betas(i) = as_scalar(xtwx_inv * xtwy);
                mat ci = xtwx_inv * xtw;
                mat si = x(i) * ci;
                shat_all(0, thread) += si(0, i);
                shat_all(1, thread) += det(si * si.t());
            }
            catch (const exception& e)
            {
                GWM_LOG_ERROR(e.what());
                flag = false;
            }
        }
    }
    shat = sum(shat_all, 1);
    return betas;
}

#endif

#ifdef ENABLE_CUDA
vec GWRMultiscale::fitVarCoreCuda(const vec &x, const vec &y, const SpatialWeight& sw, mat &S)
{
    uword nDp = mCoords.n_rows;
    mat betas(1, nDp, fill::zeros);
    mat xt = trans(x);
    cumat u_xt(xt), u_y(y);
    cumat u_betas(1, nDp);
    cumat u_dists(nDp, 1), u_weights(nDp, 1);
    custride u_xtw(1, nDp, mGroupLength);
    int *d_info, *p_info;
    p_info = new int[mGroupLength];
    checkCudaErrors(cudaMalloc(&d_info, sizeof(int) * mGroupLength));
    uword rangeSize = mWorkRange.second - mWorkRange.first;
    size_t groups = rangeSize / mGroupLength + (rangeSize % mGroupLength == 0 ? 0 : 1);
    S = mat(mHasHatMatrix ? rangeSize : 1, nDp, fill::zeros);
    if (mHasHatMatrix)
    {
        mat sg(nDp, mGroupLength, fill::zeros);
        for (size_t i = 0; i < groups; i++)
        {
            GWM_LOG_STOP_BREAK(mStatus);
            size_t begin = mWorkRange.first + i * mGroupLength, length = (begin + mGroupLength > mWorkRange.second) ? (mWorkRange.second - begin) : mGroupLength;
            for (size_t j = 0, e = begin + j; j < length; j++, e++)
            {
                checkCudaErrors(sw.weightVector(e, u_dists.dmem(), u_weights.dmem()));
                u_xtw.strides(j) = u_xt.diagmul(u_weights);
            }
            custride u_xtwx = u_xtw * u_xt.t();
            custride u_xtwy = u_xtw * u_y;
            custride u_xtwxI = u_xtwx.inv(d_info);
            checkCudaErrors(cudaMemcpy(p_info, d_info, sizeof(int) * mGroupLength, cudaMemcpyDeviceToHost));
            for (size_t j = 0; j < mGroupLength; j++)
            {
                if (p_info[j] != 0)
                {
                    std::runtime_error e("Cuda failed to get the inverse of matrix");
                    GWM_LOG_ERROR(e.what());
                    throw e;
                }
            }
            u_betas.as_stride().strides(begin, begin + length) = u_xtwxI * u_xtwy;
            custride u_c = u_xtwxI * u_xtw;
            custride u_s = u_xt.as_stride().strides(begin, begin + length).t() * u_c;
            u_s.get(sg.memptr());
            S.rows(begin, begin + length - 1) = sg.head_cols(length).t();
            GWM_LOG_PROGRESS(begin + length, nDp);
        }
    }
    else
    {
        for (size_t i = 0; i < groups; i++)
        {
            GWM_LOG_STOP_BREAK(mStatus);
            size_t begin = mWorkRange.first + i * mGroupLength, length = (begin + mGroupLength > mWorkRange.second) ? (mWorkRange.second - begin) : mGroupLength;
            for (size_t j = 0, e = begin + j; j < length; j++, e++)
            {
                checkCudaErrors(sw.weightVector(e, u_dists.dmem(), u_weights.dmem()));
                u_xtw.strides(j) = u_xt.diagmul(u_weights);
            }
            custride u_xtwx = u_xtw * u_xt.t();
            custride u_xtwy = u_xtw * u_y;
            custride u_xtwxI = u_xtwx.inv(d_info);
            checkCudaErrors(cudaMemcpy(p_info, d_info, sizeof(int) * mGroupLength, cudaMemcpyDeviceToHost));
            for (size_t j = 0; j < mGroupLength; j++)
            {
                if (p_info[j] != 0)
                {
                    std::runtime_error e("Cuda failed to get the inverse of matrix");
                    GWM_LOG_ERROR(e.what());
                    throw e;
                }
            }
            u_betas.as_stride().strides(begin, begin + length) = u_xtwxI * u_xtwy;
            GWM_LOG_PROGRESS(begin + length, nDp);
        }
        
    }
    u_betas.get(betas.memptr());
    return betas.t();
}

vec GWRMultiscale::fitVarCoreCVCuda(const vec &x, const vec &y, const SpatialWeight& sw)
{
    uword nDp = mCoords.n_rows;
    constexpr size_t nVar = 1;
    cumat u_xt(mXi.t()), u_y(mYi);
    cumat u_dists(nDp, 1), u_weights(nDp, 1);
    cumat u_betas(1, nDp);
    custride u_xtw(nVar, nDp, mGroupLength);
    int *d_info, *p_info;
    p_info = new int[mGroupLength];
    checkCudaErrors(cudaMalloc(&d_info, sizeof(int) * mGroupLength));
    bool success = true;
    uword rangeSize = mWorkRange.second - mWorkRange.first;
    size_t groups = rangeSize / mGroupLength + (rangeSize % mGroupLength == 0 ? 0 : 1);
    for (size_t i = 0; i < groups && success; i++)
    {
        GWM_LOG_STOP_BREAK(mStatus);
        size_t begin = mWorkRange.first + i * mGroupLength, length = (begin + mGroupLength > mWorkRange.second) ? (mWorkRange.second - begin) : mGroupLength;
        for (size_t j = 0, e = begin + j; j < length; j++, e++)
        {
            checkCudaErrors(sw.weightVector(e, u_dists.dmem(), u_weights.dmem()));
            checkCudaErrors(cudaMemcpy(u_weights.dmem() + e, &cubase::beta0, sizeof(double), cudaMemcpyHostToDevice));
            u_xtw.strides(j) = u_xt.diagmul(u_weights);
        }
        custride u_xtwx = u_xtw * u_xt.t();
        custride u_xtwy = u_xtw * u_y;
        custride u_xtwxI = u_xtwx.inv(d_info);
        checkCudaErrors(cudaMemcpy(p_info, d_info, sizeof(int) * mGroupLength, cudaMemcpyDeviceToHost));
        for (size_t j = 0; j < mGroupLength; j++)
        {
            if (p_info[j] != 0)
            {
                GWM_LOG_ERROR("Cuda failed to get the inverse of matrix");
                success = false;
                break;
            }
        }
        if (success)
        {
            u_betas.as_stride().strides(begin, begin + length) = u_xtwxI * u_xtwy;
        }
    }
    checkCudaErrors(cudaFree(d_info));
    delete[] p_info;
    vec betas(nDp, fill::zeros);
    u_betas.get(betas.memptr());
    return betas;
}

vec GWRMultiscale::fitVarCoreSHatCuda(const vec &x, const vec &y, const SpatialWeight& sw, vec& shat)
{
    uword nDp = mCoords.n_rows;
    constexpr size_t nVar = 1;
    cumat u_xt(mXi.t()), u_y(mYi), u_betas(nVar, nDp);
    cumat u_dists(nDp, 1), u_weights(nDp, 1);
    custride u_xtw(nVar, nDp, mGroupLength);
    mat betas(nVar, nDp);
    shat = vec(2);
    vec sst(mGroupLength);
    mat sg(nDp, mGroupLength);
    int *d_info, *p_info;
    p_info = new int[mGroupLength];
    checkCudaErrors(cudaMalloc(&d_info, sizeof(int) * mGroupLength));
    bool success = true;
    uword rangeSize = mWorkRange.second - mWorkRange.first;
    size_t groups = rangeSize / mGroupLength + (rangeSize % mGroupLength == 0 ? 0 : 1);
    for (size_t i = 0; i < groups && success; i++)
    {
        GWM_LOG_STOP_BREAK(mStatus);
        size_t begin = mWorkRange.first + i * mGroupLength, length = (begin + mGroupLength > mWorkRange.second) ? (mWorkRange.second - begin) : mGroupLength;
        for (size_t j = 0, e = begin + j; j < length; j++, e++)
        {
            checkCudaErrors(sw.weightVector(e, u_dists.dmem(), u_weights.dmem()));
            u_xtw.strides(j) = u_xt.diagmul(u_weights);
        }
        custride u_xtwx = u_xtw * u_xt.t();
        custride u_xtwy = u_xtw * u_y;
        custride u_xtwxI = u_xtwx.inv(d_info);
        checkCudaErrors(cudaMemcpy(p_info, d_info, sizeof(int) * mGroupLength, cudaMemcpyDeviceToHost));
        for (size_t j = 0; j < mGroupLength; j++)
        {
            if (p_info[j] != 0)
            {
                GWM_LOG_ERROR("Cuda failed to get the inverse of matrix");
                success = false;
                break;
            }
        }
        if (success)
        {
            u_betas.as_stride().strides(begin, begin + length) = u_xtwxI * u_xtwy;
            custride u_c = u_xtwxI * u_xtw;
            custride u_s = u_xt.as_stride().strides(begin, begin + length).t() * u_c;
            custride u_sst = u_s * u_s.t();
            u_s.get(sg.memptr());
            u_sst.get(sst.memptr());
            shat(0) += trace(sg.submat(begin, 0, arma::SizeMat(length, length)));
            shat(1) += sum(sst);
        }
    }
    checkCudaErrors(cudaFree(d_info));
    delete[] p_info;
    u_betas.get(betas.memptr());
    return betas.t();
}
#endif // ENABLE_CUDA

#ifdef ENABLE_MPI
vec GWRMultiscale::fitVarMpi(const size_t var)
{
    uword nDp = mXi.n_rows;
    mat S;
    vec beta_p = (this->*mFitVarCore)(mXi, mYi, mSpatialWeights[var], S);
    vec beta(nDp);
    MPI_Allreduce(beta_p.memptr(), beta.memptr(), nDp, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    if (mHasHatMatrix)
    {
        mat SArrayi = mSArray.slice(var) - mS0;
        mat_mul_mpi(S, SArrayi, mSArray.slice(var), mWorkerId, mWorkerNum, mWorkRangeSize);
        mSArray.slice(var) += S;
        mS0 = mSArray.slice(var) - SArrayi;
    }
    return beta;
}

double GWRMultiscale::bandwidthSizeCriterionVarCVMpi(BandwidthWeight* bandwidthWeight)
{
    SpatialWeight sw(bandwidthWeight, mSpatialWeights[mBandwidthSelectionCurrentIndex].distance());
    int status = 1;
    arma::Col<int> status_all(mWorkerNum);
    vec betas;
    try
    {
        betas = (this->*mFitVarCoreCV)(mXi, mYi, sw);
    }
    catch(const std::exception& e)
    {
        status = 0;
    }
    MPI_Allgather(&status, 1, MPI_INT, status_all.memptr(), 1, MPI_INT, MPI_COMM_WORLD);
    if (!all(status_all))
        return DBL_MAX;
    
    // If all right, calculate cv;
    double cv;
    vec betas_all;
    GWM_MPI_MASTER_BEGIN
    betas_all = vec(size(betas));
    GWM_MPI_MASTER_END
    MPI_Reduce(betas.memptr(), betas_all.memptr(), betas.n_elem, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    GWM_MPI_MASTER_BEGIN
    vec residual = mYi - mXi % betas_all;
    cv = sum(residual % residual);
    GWM_MPI_MASTER_END
    MPI_Bcast(&cv, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    if (mStatus == Status::Success && isfinite(cv))
    {
        GWM_LOG_INFO(IBandwidthSelectable::infoBandwidthCriterion(bandwidthWeight, cv));
        GWM_LOG_PROGRESS_PERCENT(exp(- abs(mBandwidthLastCriterion - cv)));
        mBandwidthLastCriterion = cv;
        return cv;
    }
    else return DBL_MAX;
}

double GWRMultiscale::bandwidthSizeCriterionVarAICMpi(BandwidthWeight* bandwidthWeight)
{
    SpatialWeight sw(bandwidthWeight, mSpatialWeights[mBandwidthSelectionCurrentIndex].distance());
    int status = 1;
    arma::Col<int> status_all(mWorkerNum);
    vec betas, shat;
    try
    {
        betas = (this->*mFitVarCoreSHat)(mXi, mYi, sw, shat);
    }
    catch(const std::exception& e)
    {
        status = 0;
    }
    MPI_Allgather(&status, 1, MPI_INT, status_all.memptr(), 1, MPI_INT, MPI_COMM_WORLD);
    if (!all(status_all))
        return DBL_MAX;
    
    // If all right, calculate cv;
    double aic;
    vec betas_all, shat_all;
    GWM_MPI_MASTER_BEGIN
    betas_all = vec(size(betas));
    shat_all = vec(size(shat));
    GWM_MPI_MASTER_END
    MPI_Reduce(betas.memptr(), betas_all.memptr(), betas.n_elem, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(shat.memptr(), shat_all.memptr(), shat.n_elem, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    GWM_MPI_MASTER_BEGIN
    aic = GWRBasic::AICc(mXi, mYi, betas_all, shat_all);
    GWM_MPI_MASTER_END
    MPI_Bcast(&aic, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    if (mStatus == Status::Success && isfinite(aic))
    {
        GWM_LOG_INFO(IBandwidthSelectable::infoBandwidthCriterion(bandwidthWeight, aic));
        GWM_LOG_PROGRESS_PERCENT(exp(- abs(mBandwidthLastCriterion - aic)));
        mBandwidthLastCriterion = aic;
        return aic;
    }
    else return DBL_MAX;
}
#endif // ENABLE_MPI

GWRMultiscale::BandwidthSizeCriterionFunction GWRMultiscale::bandwidthSizeCriterionVar(GWRMultiscale::BandwidthSelectionCriterionType type)
{
#ifdef ENABLE_MPI
    if (mParallelType & ParallelType::MPI)
    {
        switch (type)
        {
        case BandwidthSelectionCriterionType::AIC:
            return &GWRMultiscale::bandwidthSizeCriterionVarAICMpi;
        default:
            return &GWRMultiscale::bandwidthSizeCriterionVarCVMpi;
        }
    }
#endif // ENABLE_MPI
    switch (type)
    {
    case BandwidthSelectionCriterionType::AIC:
        return &GWRMultiscale::bandwidthSizeCriterionVarAICBase;
    default:
        return &GWRMultiscale::bandwidthSizeCriterionVarCVBase;
    }
}

void GWRMultiscale::setParallelType(const ParallelType &type)
{
    if (parallelAbility() & type)
    {
        mParallelType = type;
        switch (type) {
#ifdef ENABLE_OPENMP
        case ParallelType::OpenMP:
            mFitVarCore = &GWRMultiscale::fitVarCoreOmp;
            mFitVarCoreCV = &GWRMultiscale::fitVarCoreCVOmp;
            mFitVarCoreSHat = &GWRMultiscale::fitVarCoreSHatOmp;
            break;
#endif
#ifdef ENABLE_CUDA
        case ParallelType::CUDA:
            mFitVarCore = &GWRMultiscale::fitVarCoreCuda;
            mFitVarCoreCV = &GWRMultiscale::fitVarCoreCVCuda;
            mFitVarCoreSHat = &GWRMultiscale::fitVarCoreSHatCuda;
            break;
#endif
        default:
            mFitVarCore = &GWRMultiscale::fitVarCoreSerial;
            mFitVarCoreCV = &GWRMultiscale::fitVarCoreCVSerial;
            mFitVarCoreSHat = &GWRMultiscale::fitVarCoreSHatSerial;
            break;
        }
#ifdef ENABLE_MPI
        if (mParallelType & ParallelType::MPI)
        {
            mFitVar = &GWRMultiscale::fitVarMpi;
        }
        else
        {
            mFitVar = &GWRMultiscale::fitVarBase;
        }
#endif // ENABLE_MPI
    }
}

void GWRMultiscale::setSpatialWeights(const vector<SpatialWeight> &spatialWeights)
{
    SpatialMultiscaleAlgorithm::setSpatialWeights(spatialWeights);
    if (spatialWeights.size() > 0)
    {
        setInitSpatialWeight(spatialWeights[0]);
    }
}

void GWRMultiscale::setBandwidthSelectionApproach(const vector<BandwidthSelectionCriterionType> &bandwidthSelectionApproach)
{
    if (bandwidthSelectionApproach.size() == mX.n_cols)
    {
        mBandwidthSelectionApproach = bandwidthSelectionApproach;
    }
    else
    {
        length_error e("bandwidthSelectionApproach size do not match indepvars");
        GWM_LOG_ERROR(e.what());
        throw e;
    }  
}

void GWRMultiscale::setBandwidthInitilize(const vector<BandwidthInitilizeType> &bandwidthInitilize)
{
    if(bandwidthInitilize.size() == mX.n_cols){
        mBandwidthInitilize = bandwidthInitilize;
    }
    else
    {
        length_error e("BandwidthInitilize size do not match indepvars");
        GWM_LOG_ERROR(e.what());
        throw e;
    }   
}
