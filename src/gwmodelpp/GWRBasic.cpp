#include "GWRBasic.h"
#include "BandwidthSelector.h"
#include "VariableForwardSelector.h"
#include "Logger.h"

#ifdef ENABLE_OPENMP
#include <omp.h>
#endif

using namespace std;
using namespace arma;
using namespace gwm;

RegressionDiagnostic GWRBasic::CalcDiagnostic(const mat& x, const vec& y, const mat& betas, const vec& shat)
{
    vec r = y - sum(betas % x, 1);
    double rss = sum(r % r);
    double n = (double)x.n_rows;
    double AIC = n * log(rss / n) + n * log(2 * datum::pi) + n + shat(0);
    double AICc = n * log(rss / n) + n * log(2 * datum::pi) + n * ((n + shat(0)) / (n - 2 - shat(0)));
    double edf = n - 2 * shat(0) + shat(1);
    double enp = 2 * shat(0) - shat(1);
    double yss = sum((y - mean(y)) % (y - mean(y)));
    double r2 = 1 - rss / yss;
    double r2_adj = 1 - (1 - r2) * (n - 1) / (edf - 1);
    return { rss, AIC, AICc, enp, edf, r2, r2_adj };
}

mat GWRBasic::fit()
{
    GWM_LOG_STAGE("Initializing");
    uword nDp = mCoords.n_rows, nVars = mX.n_cols;
    createDistanceParameter();
    GWM_LOG_STOP_RETURN(mStatus, mat(nDp, nVars, arma::fill::zeros));

    if (mIsAutoselectIndepVars)
    {
        GWM_LOG_STAGE("Independent variable selection");
        vector<size_t> indep_vars;
        for (size_t i = (mHasIntercept ? 1 : 0); i < mX.n_cols; i++)
        {
            indep_vars.push_back(i);
        }
        size_t k = indep_vars.size();
        mIndepVarSelectionProgressTotal = (k + 1) * k / 2;
        mIndepVarSelectionProgressCurrent = 0;

        GWM_LOG_INFO(IVarialbeSelectable::infoVariableCriterion().str());
        VariableForwardSelector selector(indep_vars, mIndepVarSelectionThreshold);
        mSelectedIndepVars = selector.optimize(this);
        if (mSelectedIndepVars.size() > 0)
        {
            mX = mX.cols(VariableForwardSelector::index2uvec(mSelectedIndepVars, mHasIntercept));
            nVars = mX.n_cols;
            mIndepVarsSelectionCriterionList = selector.indepVarsCriterion();
        }
        GWM_LOG_STOP_RETURN(mStatus, mat(nDp, nVars, arma::fill::zeros));
    }

    if (mIsAutoselectBandwidth)
    {
        GWM_LOG_STAGE("Bandwidth selection");
        BandwidthWeight* bw0 = mSpatialWeight.weight<BandwidthWeight>();
        double lower = bw0->adaptive() ? 20 : 0.0;
        double upper = bw0->adaptive() ? nDp : mSpatialWeight.distance()->maxDistance();

        GWM_LOG_INFO(IBandwidthSelectable::infoBandwidthCriterion(bw0).str());
        BandwidthSelector selector(bw0, lower, upper);
        BandwidthWeight* bw = selector.optimize(this);
        if (bw)
        {
            mSpatialWeight.setWeight(bw);
            mBandwidthSelectionCriterionList = selector.bandwidthCriterion();
        }
        GWM_LOG_STOP_RETURN(mStatus, mat(nDp, nVars, arma::fill::zeros));
    }

    GWM_LOG_STAGE("Model fitting");
    mBetas = (this->*mFitFunction)(mX, mY, mBetasSE, mSHat, mQDiag, mS);
    GWM_LOG_STOP_RETURN(mStatus, mat(nDp, nVars, arma::fill::zeros));

    GWM_LOG_STAGE("Model Diagnostic");
    mDiagnostic = CalcDiagnostic(mX, mY, mBetas, mSHat);
    double trS = mSHat(0), trStS = mSHat(1);
    double sigmaHat = mDiagnostic.RSS / (nDp - 2 * trS + trStS);
    mBetasSE = sqrt(sigmaHat * mBetasSE);
    vec yhat = Fitted(mX, mBetas);
    vec res = mY - yhat;
    vec stu_res = res / sqrt(sigmaHat * mQDiag);
    mat betasTV = mBetas / mBetasSE;
    vec dybar2 = (mY - mean(mY)) % (mY - mean(mY));
    vec dyhat2 = (mY - yhat) % (mY - yhat);
    vec localR2 = vec(nDp, fill::zeros);
    for (uword i = 0; i < nDp; i++)
    {
        vec w = mSpatialWeight.weightVector(i);
        double tss = sum(dybar2 % w);
        double rss = sum(dyhat2 % w);
        localR2(i) = (tss - rss) / tss;
    }

    return mBetas;
}

mat GWRBasic::predict(const mat& locations)
{
    GWM_LOG_STAGE("Initialization");
    uword nDp = mCoords.n_rows, nVars = mX.n_cols;
    createPredictionDistanceParameter(locations);
    GWM_LOG_STOP_RETURN(mStatus, mat(nDp, nVars, arma::fill::zeros));
    
    GWM_LOG_STAGE("Prediction");
    mBetas = (this->*mPredictFunction)(locations, mX, mY);
    GWM_LOG_STOP_RETURN(mStatus, mat(nDp, nVars, arma::fill::zeros));

    return mBetas;
}

void GWRBasic::createPredictionDistanceParameter(const arma::mat& locations)
{
    if (mSpatialWeight.distance()->type() == Distance::DistanceType::CRSDistance || 
        mSpatialWeight.distance()->type() == Distance::DistanceType::MinkwoskiDistance)
    {
        mSpatialWeight.distance()->makeParameter({ locations, mCoords });
    }
}

mat GWRBasic::predictSerial(const mat& locations, const mat& x, const vec& y)
{
    uword nRp = locations.n_rows, nVar = x.n_cols;
    mat betas(nVar, nRp, fill::zeros);
    for (uword i = 0; i < nRp; i++)
    {
        GWM_LOG_STOP_BREAK(mStatus);
        vec w = mSpatialWeight.weightVector(i);
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
        GWM_LOG_PROGRESS(i + 1, nRp);
    }
    return betas.t();
}

mat GWRBasic::fitSerial(const mat& x, const vec& y, mat& betasSE, vec& shat, vec& qDiag, mat& S)
{
    uword nDp = mCoords.n_rows, nVar = x.n_cols;
    mat betas(nVar, nDp, fill::zeros);
    betasSE = mat(nVar, nDp, fill::zeros);
    shat = vec(2, fill::zeros);
    qDiag = vec(nDp, fill::zeros);
    S = mat(isStoreS() ? nDp : 1, nDp, fill::zeros);
    for (uword i = 0; i < nDp; i++)
    {
        GWM_LOG_STOP_BREAK(mStatus);
        vec w = mSpatialWeight.weightVector(i);
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
            shat(0) += si(0, i);
            shat(1) += det(si * si.t());
            vec p = - si.t();
            p(i) += 1.0;
            qDiag += p % p;
            S.row(isStoreS() ? i : 0) = si;
        }
        catch (const exception& e)
        {
            GWM_LOG_ERROR(e.what());
            throw e;
        }
        GWM_LOG_PROGRESS(i + 1, nDp);
    }
    betasSE = betasSE.t();
    return betas.t();
}

#ifdef ENABLE_OPENMP
mat GWRBasic::predictOmp(const mat& locations, const mat& x, const vec& y)
{
    uword nRp = locations.n_rows, nVar = x.n_cols;
    mat betas(nVar, nRp, arma::fill::zeros);
    bool success = true;
    std::exception except;
#pragma omp parallel for num_threads(mOmpThreadNum)
    for (int i = 0; (uword)i < nRp; i++)
    {
        GWM_LOG_STOP_CONTINUE(mStatus);
        if (success)
        {
            vec w = mSpatialWeight.weightVector(i);
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
        }
        GWM_LOG_PROGRESS(i + 1, nRp);
    }
    if (!success)
    {
        throw except;
    }
    return betas.t();
}

mat GWRBasic::fitOmp(const mat& x, const vec& y, mat& betasSE, vec& shat, vec& qDiag, mat& S)
{
    uword nDp = mCoords.n_rows, nVar = x.n_cols;
    mat betas(nVar, nDp, fill::zeros);
    betasSE = mat(nVar, nDp, fill::zeros);
    S = mat(isStoreS() ? nDp : 1, nDp, fill::zeros);
    mat shat_all(2, mOmpThreadNum, fill::zeros);
    mat qDiag_all(nDp, mOmpThreadNum, fill::zeros);
    bool success = true;
    std::exception except;
#pragma omp parallel for num_threads(mOmpThreadNum)
    for (int i = 0; (uword)i < nDp; i++)
    {
        GWM_LOG_STOP_CONTINUE(mStatus);
        if (success)
        {
            int thread = omp_get_thread_num();
            vec w = mSpatialWeight.weightVector(i);
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
                shat_all(0, thread) += si(0, i);
                shat_all(1, thread) += det(si * si.t());
                vec p = - si.t();
                p(i) += 1.0;
                qDiag_all.col(thread) += p % p;
                S.row(isStoreS() ? i : 0) = si;
            }
            catch (const exception& e)
            {
                GWM_LOG_ERROR(e.what());
                except = e;
                success = false;
            }
        }
        GWM_LOG_PROGRESS(i + 1, nDp);
    }
    if (!success)
    {
        throw except;
    }
    shat = sum(shat_all, 1);
    qDiag = sum(qDiag_all, 1);
    betasSE = betasSE.t();
    return betas.t();
}
#endif

double GWRBasic::bandwidthSizeCriterionCVSerial(BandwidthWeight* bandwidthWeight)
{
    uword nDp = mCoords.n_rows;
    vec shat(2, fill::zeros);
    double cv = 0.0;
    for (uword i = 0; i < nDp; i++)
    {
        GWM_LOG_STOP_BREAK(mStatus);
        vec d = mSpatialWeight.distance()->distance(i);
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

double GWRBasic::bandwidthSizeCriterionAICSerial(BandwidthWeight* bandwidthWeight)
{
    uword nDp = mCoords.n_rows, nVar = mX.n_cols;
    mat betas(nVar, nDp, fill::zeros);
    vec shat(2, fill::zeros);
    for (uword i = 0; i < nDp; i++)
    {
        GWM_LOG_STOP_BREAK(mStatus);
        vec d = mSpatialWeight.distance()->distance(i);
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
    double value = GWRBase::AICc(mX, mY, betas.t(), shat);
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
double GWRBasic::bandwidthSizeCriterionCVOmp(BandwidthWeight* bandwidthWeight)
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
            vec d = mSpatialWeight.distance()->distance(i);
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
                if (isfinite(res))
                    cv_all(thread) += res * res;
                else
                    flag = false;
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

double GWRBasic::bandwidthSizeCriterionAICOmp(BandwidthWeight* bandwidthWeight)
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
            vec d = mSpatialWeight.distance()->distance(i);
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
        double value = GWRBase::AICc(mX, mY, betas.t(), shat);
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

double GWRBasic::indepVarsSelectionCriterionSerial(const vector<size_t>& indepVars)
{
    mat x = mX.cols(VariableForwardSelector::index2uvec(indepVars, mHasIntercept));
    vec y = mY;
    uword nDp = mCoords.n_rows, nVar = x.n_cols;
    mat betas(nVar, nDp, fill::zeros);
    vec shat(2, fill::zeros);
    for (uword i = 0; i < nDp; i++)
    {
        GWM_LOG_STOP_BREAK(mStatus);
        vec w(nDp, fill::ones);
        mat xtw = trans(x.each_col() % w);
        mat xtwx = xtw * x;
        mat xtwy = xtw * y;
        try
        {
            mat xtwx_inv = inv_sympd(xtwx);
            betas.col(i) = xtwx_inv * xtwy;
            mat ci = xtwx_inv * xtw;
            mat si = x.row(i) * ci;
            shat(0) += si(0, i);
            shat(1) += det(si * si.t());
        }
        catch (const exception& e)
        {
            GWM_LOG_ERROR(e.what());
            return DBL_MAX;
        }
    }
    GWM_LOG_PROGRESS(++mIndepVarSelectionProgressCurrent, mIndepVarSelectionProgressTotal);
    if (mStatus == Status::Success)
    {
        double value = GWRBase::AICc(x, y, betas.t(), shat);
        GWM_LOG_INFO(IVarialbeSelectable::infoVariableCriterion(indepVars, value).str());
        return value;
    }
    else return DBL_MAX;
}

#ifdef ENABLE_OPENMP
double GWRBasic::indepVarsSelectionCriterionOmp(const vector<size_t>& indepVars)
{
    mat x = mX.cols(VariableForwardSelector::index2uvec(indepVars, mHasIntercept));
    vec y = mY;
    uword nDp = mCoords.n_rows, nVar = x.n_cols;
    mat betas(nVar, nDp, fill::zeros);
    mat shat(2, mOmpThreadNum, fill::zeros);
    int flag = true;
#pragma omp parallel for num_threads(mOmpThreadNum)
    for (int i = 0; (uword)i < nDp; i++)
    {
        GWM_LOG_STOP_CONTINUE(mStatus);
        if (flag)
        {
            int thread = omp_get_thread_num();
            vec w(nDp, fill::ones);
            mat xtw = trans(x.each_col() % w);
            mat xtwx = xtw * x;
            mat xtwy = xtw * y;
            try
            {
                mat xtwx_inv = inv_sympd(xtwx);
                betas.col(i) = xtwx_inv * xtwy;
                mat ci = xtwx_inv * xtw;
                mat si = x.row(i) * ci;
                shat(0, thread) += si(0, i);
                shat(1, thread) += det(si * si.t());
            }
            catch (const exception& e)
            {
                GWM_LOG_ERROR(e.what());
                flag = false;
            }
        }
    }
    GWM_LOG_PROGRESS(++mIndepVarSelectionProgressCurrent, mIndepVarSelectionProgressTotal);
    if (mStatus == Status::Success && flag)
    {
        double value = GWRBase::AICc(x, y, betas.t(), sum(shat, 1));
        GWM_LOG_INFO(IVarialbeSelectable::infoVariableCriterion(indepVars, value).str());
        return value;
    }
    else return DBL_MAX;
}
#endif


void GWRBasic::setBandwidthSelectionCriterion(const BandwidthSelectionCriterionType& criterion)
{
    mBandwidthSelectionCriterion = criterion;
    unordered_map<BandwidthSelectionCriterionType, BandwidthSelectionCriterionCalculator> mapper;
    switch (mParallelType)
    {
    case ParallelType::SerialOnly:
        mapper = {
            make_pair(BandwidthSelectionCriterionType::CV, &GWRBasic::bandwidthSizeCriterionCVSerial),
            make_pair(BandwidthSelectionCriterionType::AIC, &GWRBasic::bandwidthSizeCriterionAICSerial)
        };
        break;
#ifdef ENABLE_OPENMP
    case ParallelType::OpenMP:
        mapper = {
            make_pair(BandwidthSelectionCriterionType::CV, &GWRBasic::bandwidthSizeCriterionCVOmp),
            make_pair(BandwidthSelectionCriterionType::AIC, &GWRBasic::bandwidthSizeCriterionAICOmp)
        };
        break;
#endif
    default:
        mapper = {
            make_pair(BandwidthSelectionCriterionType::CV, &GWRBasic::bandwidthSizeCriterionCVSerial),
            make_pair(BandwidthSelectionCriterionType::AIC, &GWRBasic::bandwidthSizeCriterionAICSerial)
        };
        break;
    }
    mBandwidthSelectionCriterionFunction = mapper[mBandwidthSelectionCriterion];
}

void GWRBasic::setParallelType(const ParallelType& type)
{
    if (type & parallelAbility())
    {
        mParallelType = type;
        switch (type) {
        case ParallelType::SerialOnly:
            mPredictFunction = &GWRBasic::predictSerial;
            mFitFunction = &GWRBasic::fitSerial;
            mIndepVarsSelectionCriterionFunction = &GWRBasic::indepVarsSelectionCriterionSerial;
            break;
#ifdef ENABLE_OPENMP
        case ParallelType::OpenMP:
            mPredictFunction = &GWRBasic::predictOmp;
            mFitFunction = &GWRBasic::fitOmp;
            mIndepVarsSelectionCriterionFunction = &GWRBasic::indepVarsSelectionCriterionOmp;
            break;
#endif
        default:
            mPredictFunction = &GWRBasic::predictSerial;
            mFitFunction = &GWRBasic::fitSerial;
            mIndepVarsSelectionCriterionFunction = &GWRBasic::indepVarsSelectionCriterionSerial;
            break;
        }
        setBandwidthSelectionCriterion(mBandwidthSelectionCriterion);
    }
}

bool GWRBasic::isValid()
{
    if (GWRBase::isValid())
    {
        double bw = mSpatialWeight.weight<BandwidthWeight>()->bandwidth();
        if (!(bw > 0))
        {
            return false;
        }

        return true;
    }
    else return false;
}
