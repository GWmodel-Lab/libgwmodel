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

using namespace std;
using namespace arma;
using namespace gwm;

int GWRMultiscale::treeChildCount = 0;

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
    GWM_LOG_STAGE("Initializing");
    uword nDp = mX.n_rows, nVar = mX.n_cols;
    createDistanceParameter(nVar);
    createInitialDistanceParameter();
    GWM_LOG_STOP_RETURN(mStatus, mat(nDp, nVar, arma::fill::zeros));

    // ********************************
    // Centering and scaling predictors
    // ********************************
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

            GWM_LOG_INFO(string(GWM_LOG_TAG_MGWR_INITIAL_BW) + IVarialbeSelectable::infoVariableCriterion().str());
            BandwidthSelector selector;
            selector.setBandwidth(bw0);
            selector.setLower(adaptive ? mAdaptiveLower : 0.0);
            selector.setUpper(adaptive ? mCoords.n_rows : mSpatialWeights[i].distance()->maxDistance());
            BandwidthWeight* bw = selector.optimize(this);
            if (bw)
            {
                mSpatialWeights[i].setWeight(bw);
            }
            GWM_LOG_INFO((stringstream(GWM_LOG_TAG_MGWR_INITIAL_BW) << to_string(i) << "," << bw->bandwidth()).str());
        }
        GWM_LOG_STOP_RETURN(mStatus, mat(nDp, nVar, arma::fill::zeros));
    }

    // *****************************************************
    // Calculate the initial beta0 from the above bandwidths
    // *****************************************************
    GWM_LOG_STAGE("Calculating initial beta0 from initial bandwidths");
    BandwidthWeight* bw0 = bandwidth(0);
    bool adaptive = bw0->adaptive();
    mBandwidthSizeCriterion = bandwidthSizeCriterionAll(mBandwidthSelectionApproach[0]);
    GWM_LOG_STOP_RETURN(mStatus, mat(nDp, nVar, arma::fill::zeros));
    
    GWM_LOG_STAGE("Calculating initial bandwidth");
    BandwidthSelector initBwSelector;
    initBwSelector.setBandwidth(bw0);
    double maxDist = mSpatialWeights[0].distance()->maxDistance();
    initBwSelector.setLower(adaptive ? mAdaptiveLower : maxDist / 5000.0);
    initBwSelector.setUpper(adaptive ? mCoords.n_rows : maxDist);
    BandwidthWeight* initBw = initBwSelector.optimize(this);
    if (!initBw)
    {
        throw std::runtime_error("Cannot select initial bandwidth.");
    }
    mInitSpatialWeight.setWeight(initBw);
    GWM_LOG_STOP_RETURN(mStatus, mat(nDp, nVar, arma::fill::zeros));

    // 初始化诊断信息矩阵
    if (mHasHatMatrix)
    {
        mS0 = mat(nDp, nDp, fill::zeros);
        mSArray = cube(nDp, nDp, nVar, fill::zeros);
        mC = cube(nVar, nDp, nDp, fill::zeros);
    }

    GWM_LOG_STAGE("Model fitting");
    mBetas = backfitting(mX, mY);
    GWM_LOG_STOP_RETURN(mStatus, mat(nDp, nVar, arma::fill::zeros));

    // Diagnostic
    GWM_LOG_STAGE("Model Diagnostic");
    vec shat = { 
        mHasHatMatrix ? trace(mS0) : 0,
        mHasHatMatrix ? trace(mS0.t() * mS0) : 0
    };
    mDiagnostic = CalcDiagnostic(mX, mY, shat, mRSS0);
    if (mHasHatMatrix)
    {
        mBetasTV = mBetas / mBetasSE;
    }
    vec yhat = Fitted(mX, mBetas);
    vec residual = mY - yhat;

    return mBetas;
}

void GWRMultiscale::createInitialDistanceParameter()
{//回归距离计算
    if (mInitSpatialWeight.distance()->type() == Distance::DistanceType::CRSDistance || 
        mInitSpatialWeight.distance()->type() == Distance::DistanceType::MinkwoskiDistance)
    {
        mInitSpatialWeight.distance()->makeParameter({ mCoords, mCoords });
    }
}

mat GWRMultiscale::backfitting(const mat &x, const vec &y)
{
    GWM_LOG_MGWR_BACKFITTING("Model fitting with inital bandwidth");
    uword nDp = mCoords.n_rows, nVar = mX.n_cols;
    mat betas = (this->*mFitAll)(x, y);
    GWM_LOG_STOP_RETURN(mStatus, mat(nDp, nVar, arma::fill::zeros));

    mat idm = eye(nVar, nVar);
    if (mHasHatMatrix)
    {
        for (uword i = 0; i < nVar; ++i)
        {
            for (uword j = 0; j < nDp ; ++j)
            {
                mSArray.slice(i).row(j) = x(j, i) * (idm.row(i) * mC.slice(j));
            }
        }
    }

    // ***********************************************************
    // Select the optimum bandwidths for each independent variable
    // ***********************************************************
    GWM_LOG_MGWR_BACKFITTING("Selecting the optimum bandwidths for each independent variable");
    uvec bwChangeNo(nVar, fill::zeros);
    vec resid = y - Fitted(x, betas);
    double RSS0 = sum(resid % resid), RSS1 = DBL_MAX;
    double criterion = DBL_MAX;
    for (size_t iteration = 1; iteration <= mMaxIteration && criterion > mCriterionThreshold; iteration++)
    {
        GWM_LOG_STOP_BREAK(mStatus);
        GWM_LOG_MGWR_BACKFITTING("#iteration " + to_string(iteration));
        for (uword i = 0; i < nVar  ; i++)
        {
            GWM_LOG_STOP_BREAK(mStatus);
            vec fi = betas.col(i) % x.col(i);
            vec yi = resid + fi;
            if (mBandwidthInitilize[i] != BandwidthInitilizeType::Specified)
            {
                GWM_LOG_MGWR_BACKFITTING("#variable-bandwidth-selection " + to_string(i));
                mBandwidthSizeCriterion = bandwidthSizeCriterionVar(mBandwidthSelectionApproach[i]);
                mBandwidthSelectionCurrentIndex = i;
                mYi = yi;
                mXi = mX.col(i);
                BandwidthWeight* bwi0 = bandwidth(i);
                bool adaptive = bwi0->adaptive();
                BandwidthSelector selector;
                selector.setBandwidth(bwi0);
                double maxDist = mSpatialWeights[i].distance()->maxDistance();
                selector.setLower(adaptive ? mAdaptiveLower : maxDist / 5000.0);
                selector.setUpper(adaptive ? mCoords.n_rows : maxDist);
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
                GWM_LOG_MGWR_BACKFITTING("#variable-bandwidth-selection " + strjoin(",", vbs_args));
            }
            GWM_LOG_STOP_BREAK(mStatus);

            mat S;
            betas.col(i) = (this->*mFitVar)(x.col(i), yi, i, S);
            if (mHasHatMatrix)
            {
                mat SArrayi = mSArray.slice(i) - mS0;
                mSArray.slice(i) = S * SArrayi + S;
                mS0 = mSArray.slice(i) - SArrayi;
            }
            resid = y - Fitted(x, betas);
        }
        RSS1 = RSS(x, y, betas);
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

mat GWRMultiscale::fitAllSerial(const mat& x, const vec& y)
{
    uword nDp = mCoords.n_rows, nVar = x.n_cols;
    mat betas(nVar, nDp, fill::zeros);
    if (mHasHatMatrix )
    {
        mat betasSE(nVar, nDp, fill::zeros);
        for (uword i = 0; i < nDp ; i++)
        {
            GWM_LOG_STOP_BREAK(mStatus);
            vec w = mInitSpatialWeight.weightVector(i);
            mat xtw = trans(x.each_col() % w);
            mat xtwx = xtw * x;
            mat xtwy = xtw * y;
            try
            {
                mat xtwx_inv = inv_sympd(xtwx);
                betas.col(i) = xtwx_inv * xtwy;
                mat ci = xtwx_inv * xtw;
                betasSE.col(i) = sum(ci % ci, 1);
                mat si = x.row(i) * ci;
                mS0.row(i) = si;
                mC.slice(i) = ci;
            }
            catch (const exception& e)
            {
                GWM_LOG_ERROR(e.what());
                throw e;
            }
            GWM_LOG_PROGRESS(i + 1, nDp);
        }
        mBetasSE = betasSE.t();
    }
    else
    {
        for (int i = 0; (uword)i < nDp ; i++)
        {
            GWM_LOG_STOP_BREAK(mStatus);
            vec w = mInitSpatialWeight.weightVector(i);
            mat xtw = trans(x.each_col() % w);
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
                throw e;
            }
            GWM_LOG_PROGRESS(i + 1, nDp);
        }
    }
    return betas.t();
}

#ifdef ENABLE_OPENMP
mat GWRMultiscale::fitAllOmp(const mat &x, const vec &y)
{
    uword nDp = mCoords.n_rows, nVar = x.n_cols;
    mat betas(nVar, nDp, fill::zeros);
    bool success = true;
    std::exception except;
    if (mHasHatMatrix )
    {
        mat betasSE(nVar, nDp, fill::zeros);
#pragma omp parallel for num_threads(mOmpThreadNum)
        for (int i = 0; (uword)i < nDp; i++)
        {
            GWM_LOG_STOP_CONTINUE(mStatus);
            vec w = mInitSpatialWeight.weightVector(i);
            mat xtw = trans(x.each_col() % w);
            mat xtwx = xtw * x;
            mat xtwy = xtw * y;
            try
            {
                mat xtwx_inv = inv_sympd(xtwx);
                betas.col(i) = xtwx_inv * xtwy;
                mat ci = xtwx_inv * xtw;
                betasSE.col(i) = sum(ci % ci, 1);
                mat si = x.row(i) * ci;
                mS0.row(i) = si;
                mC.slice(i) = ci;
            }
            catch (const exception& e)
            {
                GWM_LOG_ERROR(e.what());
                except = e;
                success = false;
            }
            GWM_LOG_PROGRESS(i + 1, nDp);
        }
        mBetasSE = betasSE.t();
    }
    else
    {
#pragma omp parallel for num_threads(mOmpThreadNum)
        for (int i = 0; (uword)i < nDp; i++)
        {
            GWM_LOG_STOP_CONTINUE(mStatus);
            vec w = mInitSpatialWeight.weightVector(i);
            mat xtw = trans(x.each_col() % w);
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
#endif

vec GWRMultiscale::fitVarSerial(const vec &x, const vec &y, const uword var, mat &S)
{
    uword nDp = mCoords.n_rows;
    mat betas(1, nDp, fill::zeros);
    bool success = true;
    std::exception except;
    if (mHasHatMatrix)
    {
        mat ci, si;
        S = mat(mHasHatMatrix ? nDp : 1, nDp, fill::zeros);
        for (uword i = 0; i < nDp  ; i++)
        {
            GWM_LOG_STOP_BREAK(mStatus);
            vec w = mSpatialWeights[var].weightVector(i);
            mat xtw = trans(x % w);
            mat xtwx = xtw * x;
            mat xtwy = xtw * y;
            try
            {
                mat xtwx_inv = inv_sympd(xtwx);
                betas.col(i) = xtwx_inv * xtwy;
                ci = xtwx_inv * xtw;
                si = x(i) * ci;
                S.row(mHasHatMatrix ? i : 0) = si;
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
        for (int i = 0; (uword)i < nDp  ; i++)
        {
            GWM_LOG_STOP_BREAK(mStatus);
            vec w = mSpatialWeights[var].weightVector(i);
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

#ifdef ENABLE_OPENMP
vec GWRMultiscale::fitVarOmp(const vec &x, const vec &y, const uword var, mat &S)
{
    uword nDp = mCoords.n_rows;
    mat betas(1, nDp, fill::zeros);
    bool success = true;
    std::exception except;
    if (mHasHatMatrix)
    {
        S = mat(mHasHatMatrix ? nDp : 1, nDp, fill::zeros);
#pragma omp parallel for num_threads(mOmpThreadNum)
        for (int i = 0; (uword)i < nDp; i++)
        {
            GWM_LOG_STOP_CONTINUE(mStatus);
            vec w = mSpatialWeights[var].weightVector(i);
            mat xtw = trans(x % w);
            mat xtwx = xtw * x;
            mat xtwy = xtw * y;
            try
            {
                mat xtwx_inv = inv_sympd(xtwx);
                betas.col(i) = xtwx_inv * xtwy;
                mat ci = xtwx_inv * xtw;
                mat si = x(i) * ci;
                S.row(mHasHatMatrix ? i : 0) = si;
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
        for (int i = 0; (uword)i < nDp; i++)
        {
            GWM_LOG_STOP_CONTINUE(mStatus);
            vec w = mSpatialWeights[var].weightVector(i);
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
#endif

double GWRMultiscale::bandwidthSizeCriterionAllCVSerial(BandwidthWeight *bandwidthWeight)
{
    uword nDp = mCoords.n_rows;
    vec shat(2, fill::zeros);
    double cv = 0.0;
    for (uword i = 0; i < nDp; i++)
    {
        GWM_LOG_STOP_BREAK(mStatus);
        vec d = mInitSpatialWeight.distance()->distance(i);
        vec w = bandwidthWeight->weight(d);
        w(i) = 0.0;
        mat xtw = trans(mX.each_col() % w);
        mat xtwx = xtw * mX;
        mat xtwy = xtw * mY;
        try
        {
            mat xtwx_inv = inv_sympd(xtwx);
            vec beta = xtwx_inv * xtwy;
            double res = mY(i) - det(mX.row(i) * beta);
            cv += res * res;
        }
        catch (const exception& e)
        {
            GWM_LOG_ERROR(e.what());
            return DBL_MAX;
        }
    }
    if (mStatus == Status::Success && isfinite(cv))
    {
        GWM_LOG_INFO(IBandwidthSelectable::infoBandwidthCriterion(bandwidthWeight, cv).str());
        GWM_LOG_PROGRESS_PERCENT(exp(- abs(mBandwidthLastCriterion - cv)));
        mBandwidthLastCriterion = cv;
        return cv;
    }
    else return DBL_MAX;
}

#ifdef ENABLE_OPENMP
double GWRMultiscale::bandwidthSizeCriterionAllCVOmp(BandwidthWeight *bandwidthWeight)
{
    uword nDp = mCoords.n_rows;
    vec shat(2, fill::zeros);
    vec cv_all(mOmpThreadNum, fill::zeros);
    bool flag = true;
#pragma omp parallel for num_threads(mOmpThreadNum)
    for (int i = 0; (uword)i < nDp; i++)
    {
        GWM_LOG_STOP_CONTINUE(mStatus);
        if (flag)
        {
            int thread = omp_get_thread_num();
            vec d = mInitSpatialWeight.distance()->distance(i);
            vec w = bandwidthWeight->weight(d);
            w(i) = 0.0;
            mat xtw = trans(mX.each_col() % w);
            mat xtwx = xtw * mX;
            mat xtwy = xtw * mY;
            try
            {
                mat xtwx_inv = inv_sympd(xtwx);
                vec beta = xtwx_inv * xtwy;
                double res = mY(i) - det(mX.row(i) * beta);
                cv_all(thread) += res * res;
            }
            catch (const exception& e)
            {
                GWM_LOG_ERROR(e.what());
                flag = false;
            }
        }
    }
    if (mStatus == Status::Success && flag)
    {
        double cv = sum(cv_all);
        GWM_LOG_INFO(IBandwidthSelectable::infoBandwidthCriterion(bandwidthWeight, cv).str());
        GWM_LOG_PROGRESS_PERCENT(exp(- abs(mBandwidthLastCriterion - cv)));
        mBandwidthLastCriterion = cv;
        return cv;
    }
    else return DBL_MAX;
}
#endif

double GWRMultiscale::bandwidthSizeCriterionAllAICSerial(BandwidthWeight *bandwidthWeight)
{
    uword nDp = mCoords.n_rows, nVar = mX.n_cols;
    mat betas(nVar, nDp, fill::zeros);
    vec shat(2, fill::zeros);
    for (uword i = 0; i < nDp ; i++)
    {
        GWM_LOG_STOP_BREAK(mStatus);
        vec d = mInitSpatialWeight.distance()->distance(i);
        vec w = bandwidthWeight->weight(d);
        mat xtw = trans(mX.each_col() % w);
        mat xtwx = xtw * mX;
        mat xtwy = xtw * mY;
        try
        {
            mat xtwx_inv = inv_sympd(xtwx);
            betas.col(i) = xtwx_inv * xtwy;
            mat ci = xtwx_inv * xtw;
            mat si = mX.row(i) * ci;
            shat(0) += si(0, i);
            shat(1) += det(si * si.t());
        }
        catch (const exception& e)
        {
            GWM_LOG_ERROR(e.what());
            return DBL_MAX;
        }
    }
    double value = GWRMultiscale::AICc(mX, mY, betas.t(), shat);
    if (mStatus == Status::Success && isfinite(value))
    {
        GWM_LOG_INFO(IBandwidthSelectable::infoBandwidthCriterion(bandwidthWeight, value).str());
        GWM_LOG_PROGRESS_PERCENT(exp(- abs(mBandwidthLastCriterion - value)));
        mBandwidthLastCriterion = value;
        return value;
    }
    else return DBL_MAX;
}

#ifdef ENABLE_OPENMP
double GWRMultiscale::bandwidthSizeCriterionAllAICOmp(BandwidthWeight *bandwidthWeight)
{
    uword nDp = mCoords.n_rows, nVar = mX.n_cols;
    mat betas(nVar, nDp, fill::zeros);
    mat shat_all(2, mOmpThreadNum, fill::zeros);
    bool flag = true;
#pragma omp parallel for num_threads(mOmpThreadNum)
    for (int i = 0; (uword)i < nDp; i++)
    {
        GWM_LOG_STOP_CONTINUE(mStatus);
        if (flag)
        {
            int thread = omp_get_thread_num();
            vec d = mInitSpatialWeight.distance()->distance(i);
            vec w = bandwidthWeight->weight(d);
            mat xtw = trans(mX.each_col() % w);
            mat xtwx = xtw * mX;
            mat xtwy = xtw * mY;
            try
            {
                mat xtwx_inv = inv_sympd(xtwx);
                betas.col(i) = xtwx_inv * xtwy;
                mat ci = xtwx_inv * xtw;
                mat si = mX.row(i) * ci;
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
    if (mStatus == Status::Success && flag)
    {
        vec shat = sum(shat_all, 1);
        double value = GWRMultiscale::AICc(mX, mY, betas.t(), shat);
        if (isfinite(value))
        {
            GWM_LOG_INFO(IBandwidthSelectable::infoBandwidthCriterion(bandwidthWeight, value).str());
            GWM_LOG_PROGRESS_PERCENT(exp(- abs(mBandwidthLastCriterion - value)));
            mBandwidthLastCriterion = value;
            return value;
        }
        else return DBL_MAX;
    }
    else return DBL_MAX;
}
#endif

double GWRMultiscale::bandwidthSizeCriterionVarCVSerial(BandwidthWeight *bandwidthWeight)
{
    size_t var = mBandwidthSelectionCurrentIndex;
    uword nDp = mCoords.n_rows;
    vec shat(2, fill::zeros);
    double cv = 0.0;
    for (uword i = 0; i < nDp; i++)
    {
        GWM_LOG_STOP_BREAK(mStatus);
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
            cv += res * res;
        }
        catch (const exception& e)
        {
            GWM_LOG_ERROR(e.what());
            return DBL_MAX;
        }
    }
    if (mStatus == Status::Success && isfinite(cv))
    {
        GWM_LOG_INFO(IBandwidthSelectable::infoBandwidthCriterion(bandwidthWeight, cv).str());
        GWM_LOG_PROGRESS_PERCENT(exp(- abs(mBandwidthLastCriterion - cv)));
        mBandwidthLastCriterion = cv;
        return cv;
    }
    else return DBL_MAX;
}

#ifdef ENABLE_OPENMP
double GWRMultiscale::bandwidthSizeCriterionVarCVOmp(BandwidthWeight *bandwidthWeight)
{
    size_t var = mBandwidthSelectionCurrentIndex;
    uword nDp = mCoords.n_rows;
    vec shat(2, fill::zeros);
    vec cv_all(mOmpThreadNum, fill::zeros);
    bool flag = true;
#pragma omp parallel for num_threads(mOmpThreadNum)
    for (int i = 0; (uword)i < nDp; i++)
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
                cv_all(thread) += res * res;
            }
            catch (const exception& e)
            {
                GWM_LOG_ERROR(e.what());
                flag = false;
            }
        }
    }
    if (mStatus == Status::Success && flag)
    {
        double cv = sum(cv_all);
        GWM_LOG_INFO(IBandwidthSelectable::infoBandwidthCriterion(bandwidthWeight, cv).str());
        GWM_LOG_PROGRESS_PERCENT(exp(- abs(mBandwidthLastCriterion - cv)));
        mBandwidthLastCriterion = cv;
        return cv;
    }
    else return DBL_MAX;
}
#endif

double GWRMultiscale::bandwidthSizeCriterionVarAICSerial(BandwidthWeight *bandwidthWeight)
{
    size_t var = mBandwidthSelectionCurrentIndex;
    uword nDp = mCoords.n_rows;
    mat betas(1, nDp, fill::zeros);
    vec shat(2, fill::zeros);
    for (uword i = 0; i < nDp ; i++)
    {
        GWM_LOG_STOP_BREAK(mStatus);
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
            GWM_LOG_ERROR(e.what());
            return DBL_MAX;
        }
    }
    double value = GWRMultiscale::AICc(mXi, mYi, betas.t(), shat);
    if (mStatus == Status::Success && isfinite(value))
    {
        GWM_LOG_INFO(IBandwidthSelectable::infoBandwidthCriterion(bandwidthWeight, value).str());
        GWM_LOG_PROGRESS_PERCENT(exp(- abs(mBandwidthLastCriterion - value)));
        mBandwidthLastCriterion = value;
        return value;
    }
    return isfinite(value) ? value : DBL_MAX;
}

#ifdef ENABLE_OPENMP
double GWRMultiscale::bandwidthSizeCriterionVarAICOmp(BandwidthWeight *bandwidthWeight)
{
    size_t var = mBandwidthSelectionCurrentIndex;
    uword nDp = mCoords.n_rows;
    mat betas(1, nDp, fill::zeros);
    mat shat_all(2, mOmpThreadNum, fill::zeros);
    bool flag = true;
#pragma omp parallel for num_threads(mOmpThreadNum)
    for (int i = 0; (uword)i < nDp; i++)
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
                mat si = mXi(i) * ci;
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
    if (flag)
    {
        vec shat = sum(shat_all, 1);
        double value = GWRMultiscale::AICc(mXi, mYi, betas.t(), shat);
        if (isfinite(value))
        {
            GWM_LOG_INFO(IBandwidthSelectable::infoBandwidthCriterion(bandwidthWeight, value).str());
            GWM_LOG_PROGRESS_PERCENT(exp(- abs(mBandwidthLastCriterion - value)));
            mBandwidthLastCriterion = value;
            return value;
        }
        else return DBL_MAX;
    }
    return DBL_MAX;
}
#endif

GWRMultiscale::BandwidthSizeCriterionFunction GWRMultiscale::bandwidthSizeCriterionAll(GWRMultiscale::BandwidthSelectionCriterionType type)
{
    unordered_map<BandwidthSelectionCriterionType, unordered_map<ParallelType, BandwidthSizeCriterionFunction> > mapper = {
        std::make_pair<BandwidthSelectionCriterionType, unordered_map<ParallelType, BandwidthSizeCriterionFunction> >(BandwidthSelectionCriterionType::CV, {
            std::make_pair(ParallelType::SerialOnly, &GWRMultiscale::bandwidthSizeCriterionAllCVSerial),
        #ifdef ENABLE_OPENMP
            std::make_pair(ParallelType::OpenMP, &GWRMultiscale::bandwidthSizeCriterionAllCVOmp),
        #endif
            std::make_pair(ParallelType::CUDA, &GWRMultiscale::bandwidthSizeCriterionAllCVSerial)
        }),
        std::make_pair<BandwidthSelectionCriterionType, unordered_map<ParallelType, BandwidthSizeCriterionFunction> >(BandwidthSelectionCriterionType::AIC, {
            std::make_pair(ParallelType::SerialOnly, &GWRMultiscale::bandwidthSizeCriterionAllAICSerial),
        #ifdef ENABLE_OPENMP
            std::make_pair(ParallelType::OpenMP, &GWRMultiscale::bandwidthSizeCriterionAllAICOmp),
        #endif
            std::make_pair(ParallelType::CUDA, &GWRMultiscale::bandwidthSizeCriterionAllAICSerial)
        })
    };
    return mapper[type][mParallelType];
}

GWRMultiscale::BandwidthSizeCriterionFunction GWRMultiscale::bandwidthSizeCriterionVar(GWRMultiscale::BandwidthSelectionCriterionType type)
{
    unordered_map<BandwidthSelectionCriterionType, unordered_map<ParallelType, BandwidthSizeCriterionFunction> > mapper = {
        std::make_pair<BandwidthSelectionCriterionType, unordered_map<ParallelType, BandwidthSizeCriterionFunction> >(BandwidthSelectionCriterionType::CV, {
            std::make_pair(ParallelType::SerialOnly, &GWRMultiscale::bandwidthSizeCriterionVarCVSerial),
        #ifdef ENABLE_OPENMP
            std::make_pair(ParallelType::OpenMP, &GWRMultiscale::bandwidthSizeCriterionVarCVOmp),
        #endif
            std::make_pair(ParallelType::CUDA, &GWRMultiscale::bandwidthSizeCriterionVarCVSerial)
        }),
        std::make_pair<BandwidthSelectionCriterionType, unordered_map<ParallelType, BandwidthSizeCriterionFunction> >(BandwidthSelectionCriterionType::AIC, {
            std::make_pair(ParallelType::SerialOnly, &GWRMultiscale::bandwidthSizeCriterionVarAICSerial),
        #ifdef ENABLE_OPENMP
            std::make_pair(ParallelType::OpenMP, &GWRMultiscale::bandwidthSizeCriterionVarAICOmp),
        #endif
            std::make_pair(ParallelType::CUDA, &GWRMultiscale::bandwidthSizeCriterionVarAICSerial)
        })
    };
    return mapper[type][mParallelType];
}

void GWRMultiscale::setParallelType(const ParallelType &type)
{
    if (parallelAbility() & type)
    {
        mParallelType = type;
        switch (type) {
        case ParallelType::SerialOnly:
            mFitAll = &GWRMultiscale::fitAllSerial;
            mFitVar = &GWRMultiscale::fitVarSerial;
            break;
#ifdef ENABLE_OPENMP
        case ParallelType::OpenMP:
            mFitAll = &GWRMultiscale::fitAllOmp;
            mFitVar = &GWRMultiscale::fitVarOmp;
            break;
#endif
        default:
            mFitAll = &GWRMultiscale::fitAllSerial;
            mFitVar = &GWRMultiscale::fitVarSerial;
            break;
        }
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
