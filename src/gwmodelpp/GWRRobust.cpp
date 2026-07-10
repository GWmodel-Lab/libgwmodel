#include "GWRRobust.h"
#include "BandwidthSelector.h"
#include "VariableForwardSelector.h"
#include <assert.h>
#include "Logger.h"

#ifdef ENABLE_OPENMP
#include <omp.h>
#endif

using namespace std;
using namespace arma;
using namespace gwm;

RegressionDiagnostic GWRRobust::CalcDiagnostic(const mat &x, const vec &y, const mat &betas, const vec &shat)
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
    return {rss, AIC, AICc, enp, edf, r2, r2_adj};
}

mat GWRRobust::fit()
{
    GWM_LOG_STAGE("Initializing");
    uword nDp = mCoords.n_rows, nVar = mX.n_cols;
    mWorkRange = make_pair(uword(0), nDp);
    createDistanceParameter();
    GWM_LOG_STOP_RETURN(mStatus, mat(nDp, nVar, arma::fill::zeros));

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

        GWM_LOG_INFO(IVarialbeSelectable::infoVariableCriterion());
        VariableForwardSelector selector(indep_vars, mIndepVarSelectionThreshold);
        mSelectedIndepVars = selector.optimize(this);
        if (mSelectedIndepVars.size() > 0)
        {
            mX = mX.cols(VariableForwardSelector::index2uvec(mSelectedIndepVars, mHasIntercept));
            nVar = mX.n_cols;
            mIndepVarsSelectionCriterionList = selector.indepVarsCriterion();
        }
        GWM_LOG_STOP_RETURN(mStatus, mat(nDp, nVar, arma::fill::zeros));
    }

    if (mIsAutoselectBandwidth)
    {
        GWM_LOG_STAGE("Bandwidth selection");
        BandwidthWeight* bw0 = mSpatialWeight.weight<BandwidthWeight>();
        double lower = bw0->adaptive() ? 20 : 0.0;
        double upper = bw0->adaptive() ? nDp : mSpatialWeight.distance()->maxDistance();
        
        GWM_LOG_INFO(IBandwidthSelectable::infoBandwidthCriterion(bw0));
        BandwidthSelector selector(bw0, lower, upper);
        BandwidthWeight* bw = selector.optimize(this);
        if (bw)
        {
            mSpatialWeight.setWeight(bw);
            mBandwidthSelectionCriterionList = selector.bandwidthCriterion();
        }
        GWM_LOG_STOP_RETURN(mStatus, mat(nDp, nVar, arma::fill::zeros));
    }

    GWM_LOG_STAGE("Model fitting");
    mWeightMask = vec(nDp, fill::ones);
    mBetas = regressionHatmatrix(mX, mY, mBetasSE, mSHat, mQDiag, mS);
    GWM_LOG_STOP_RETURN(mStatus, mat(nDp, nVar, arma::fill::zeros));

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

mat GWRRobust::predict(const mat& locations)
{
    size_t nDp = locations.n_rows, nVar = mX.n_cols;

    createPredictionDistanceParameter(locations);
    GWM_LOG_STOP_RETURN(mStatus, mat(nDp, nVar, arma::fill::zeros));

    mBetas = (this->*mPredictFunction)(locations, mX, mY);
    GWM_LOG_STOP_RETURN(mStatus, mat(nDp, nVar, arma::fill::zeros));
    
    return mBetas;
}

void GWRRobust::createPredictionDistanceParameter(const arma::mat& locations)
{
    if (mSpatialWeight.distance()->type() == Distance::DistanceType::CRSDistance || 
        mSpatialWeight.distance()->type() == Distance::DistanceType::MinkwoskiDistance)
    {
        mSpatialWeight.distance()->makeParameter({ locations, mCoords });
    }
}

mat GWRRobust::fitSerial(const mat& x, const vec& y, mat& betasSE, vec& shat, vec& qDiag, mat& S)
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
        vec w = mSpatialWeight.weightVector(i) % mWeightMask;
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
mat GWRRobust::fitOmp(const mat& x, const vec& y, mat& betasSE, vec& shat, vec& qDiag, mat& S)
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
    for (int i = 0; i < (int)nDp; i++)
    {
        GWM_LOG_STOP_CONTINUE(mStatus);
        if (success)
        {
            int thread = omp_get_thread_num();
            vec w = mSpatialWeight.weightVector(i) % mWeightMask;
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


mat GWRRobust::regressionHatmatrix(const mat &x, const vec &y, mat &betasSE, vec &shat, vec &qdiag, mat &S)
{
    if (mFiltered)
    {
        return robustGWRCaliFirst(x, y, betasSE, shat, qdiag, S);
    }
    else
    {
        return robustGWRCaliSecond(x, y, betasSE, shat, qdiag, S);
    }
}

mat GWRRobust::robustGWRCaliFirst(const mat &x, const vec &y, mat &betasSE, vec &shat, vec &qDiag, mat &S)
{
    uword nDp = x.n_rows, nVar = x.n_cols;    
    mat betas = (this->*mfitFunction)(x, y, betasSE, shat, qDiag, S);
    GWM_LOG_STOP_RETURN(mStatus, mat(nDp, nVar, arma::fill::zeros));

    //  ------------- 计算W.vect
    // vec yhat = fitted(x, betas);
    vec yhat = sum(betas % x, 1);
    vec residual = y - yhat;
    // 诊断信息
    RegressionDiagnostic diagnostic;
    diagnostic = CalcDiagnostic(x, y, betas, shat);
    double trS = shat(0), trStS = shat(1);
    double sigmaHat = diagnostic.RSS / (1.0 * nDp - 2 * trS + trStS);
    vec studentizedResidual = residual / sqrt(sigmaHat * qDiag);

    vec WVect(nDp, fill::zeros);
    //mDiagnostic = diagnostic;
    //生成W.vect
    for (int i = 0; (uword)i < studentizedResidual.size(); i++)
    {
        if (fabs(studentizedResidual[i]) > 3)
        {
            WVect(i) = 0;
        }
        else
        {
            WVect(i) = 1;
        }
    }
    mWeightMask = WVect;
    betas = (this->*mfitFunction)(x, y, betasSE, shat, qDiag, S);
    GWM_LOG_STOP_RETURN(mStatus, mat(nDp, nVar, arma::fill::zeros));

    mSHat=shat;
    return betas;
}

mat GWRRobust::robustGWRCaliSecond(const mat &x, const vec &y, mat &betasSE, vec &shat, vec &qDiag, mat &S)
{
    uword nDp = x.n_rows, nVar = x.n_cols;
    double iter = 0;
    double diffmse = 1;
    double delta = 1.0e-5;
    double maxiter = 20;
    mat betas = (this->*mfitFunction)(x, y, betasSE, shat, qDiag, S);
    GWM_LOG_STOP_RETURN(mStatus, betas);

    //计算residual
    // vec yHat = fitted(x, betas);
    vec yHat = sum(betas % x, 1);
    vec residual = y - yHat;
    //计算mse
    double mse = sum((residual % residual)) / residual.size();
    //计算WVect
    mWeightMask = filtWeight(residual, mse);
    //mDiagnostic = CalcDiagnostic(x, y, betas, shat);
    while (diffmse > delta && iter < maxiter)
    {
        double oldmse = mse;
        betas = (this->*mfitFunction)(x, y, betasSE, shat, qDiag, S);
        GWM_LOG_STOP_BREAK(mStatus);
        //计算residual
        // yHat = fitted(x, betas);
        yHat = sum(betas % x, 1);
        residual = y - yHat;
        mse = sum((residual % residual)) / residual.size();
        mWeightMask = filtWeight(residual, mse);
        diffmse = abs(oldmse - mse) / mse;
        iter = iter + 1;
    }
    GWM_LOG_STOP_RETURN(mStatus, mat(nDp, nVar, arma::fill::zeros));
    mSHat=shat;
    return betas;
}

void GWRRobust::setParallelType(const ParallelType &type)
{
    if (type & parallelAbility())
    {
        mParallelType = type;
        switch (type)
        {
        case ParallelType::SerialOnly:
            mfitFunction = &GWRRobust::fitSerial;
            break;
#ifdef ENABLE_OPENMP
        case ParallelType::OpenMP:
            mfitFunction = &GWRRobust::fitOmp;
            break;
#endif
        default:
            mfitFunction = &GWRRobust::fitSerial;
            break;
        }
    }
}

vec GWRRobust::filtWeight(vec residual, double mse)
{
    //计算residual
    vec r = abs(residual / sqrt(mse));
    vec wvect(r.size(), fill::ones);
    //数组赋值
    for (uword i = 0; i < r.size(); i++)
    {
        if (r[i] <= 2)
        {
            wvect[i] = 1;
        }
        else if (r[i] > 2 && r[i] < 3)
        {
            double f = r[i] - 2;
            wvect[i] = (1.0 - (f * f)) * (1.0 - (f * f));
        }
        else
        {
            wvect[i] = 0;
        }
    }
    return wvect;
}