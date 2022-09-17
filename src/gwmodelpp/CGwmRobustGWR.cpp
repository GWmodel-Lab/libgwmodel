#include "CGwmRobustGWR.h"
#include "CGwmBandwidthSelector.h"
#include "CGwmVariableForwardSelector.h"
#include <assert.h>

#ifdef ENABLE_OPENMP
#include <omp.h>
#endif

using namespace arma;

GwmRegressionDiagnostic CGwmRobustGWR::CalcDiagnostic(const mat &x, const vec &y, const mat &betas, const vec &shat)
{
    vec r = y - sum(betas % x, 1);
    double rss = sum(r % r);
    int n = x.n_rows;
    double AIC = n * log(rss / n) + n * log(2 * datum::pi) + n + shat(0);
    double AICc = n * log(rss / n) + n * log(2 * datum::pi) + n * ((n + shat(0)) / (n - 2 - shat(0)));
    double edf = n - 2 * shat(0) + shat(1);
    double enp = 2 * shat(0) - shat(1);
    double yss = sum((y - mean(y)) % (y - mean(y)));
    double r2 = 1 - rss / yss;
    double r2_adj = 1 - (1 - r2) * (n - 1) / (edf - 1);
    return {rss, AIC, AICc, enp, edf, r2, r2_adj};
}

CGwmRobustGWR::CGwmRobustGWR()
{
}

CGwmRobustGWR::~CGwmRobustGWR()
{
}

void CGwmRobustGWR::run()
{
    createRegressionDistanceParameter();
    assert(mRegressionDistanceParameter != nullptr);

    if (!hasPredictLayer() && mIsAutoselectIndepVars)
    {
        CGwmVariableForwardSelector selector(mIndepVars, mIndepVarSelectionThreshold);
        vector<GwmVariable> selectedIndepVars = selector.optimize(this);
        if (selectedIndepVars.size() > 0)
        {
            mIndepVars = selectedIndepVars;
            mIndepVarsSelectionCriterionList = selector.indepVarsCriterion();
        }
    }

    setXY(mX, mY, mSourceLayer, mDepVar, mIndepVars);
    uword nDp = mSourceLayer->featureCount();
    mWeightMask = vec(nDp, fill::ones);
    if (!hasPredictLayer() && mIsAutoselectBandwidth)
    {
        CGwmBandwidthWeight *bw0 = mSpatialWeight.weight<CGwmBandwidthWeight>();
        double lower = bw0->adaptive() ? 20 : 0.0;
        double upper = bw0->adaptive() ? nDp : mSpatialWeight.distance()->maxDistance(nDp, mRegressionDistanceParameter);
        CGwmBandwidthSelector selector(bw0, lower, upper);
        CGwmBandwidthWeight *bw = selector.optimize(this);
        if (bw)
        {
            mSpatialWeight.setWeight(bw);
            mBandwidthSelectionCriterionList = selector.bandwidthCriterion();
        }
    }
    if (mHasHatMatrix)
    {
        mat betasSE, S;
        vec shat, qdiag;
        mBetas = regressionHatmatrix(mX, mY, betasSE, shat, qdiag, S);
        mDiagnostic = CalcDiagnostic(mX, mY, mBetas, shat);
        double trS = shat(0), trStS = shat(1);
        double sigmaHat = mDiagnostic.RSS / (nDp - 2 * trS + trStS);
        betasSE = sqrt(sigmaHat * betasSE);
        vec yhat = Fitted(mX, mBetas);
        vec res = mY - yhat;
        vec stu_res = res / sqrt(sigmaHat * qdiag);
        mat betasTV = mBetas / betasSE;
        vec dybar2 = (mY - mean(mY)) % (mY - mean(mY));
        vec dyhat2 = (mY - yhat) % (mY - yhat);
        vec localR2 = vec(nDp, fill::zeros);
        for (uword i = 0; i < nDp; i++)
        {
            vec w = mSpatialWeight.weightVector(mRegressionDistanceParameter, i);
            double tss = sum(dybar2 % w);
            double rss = sum(dyhat2 % w);
            localR2(i) = (tss - rss) / tss;
        }
        createResultLayer({make_tuple(string("%1"), mBetas, NameFormat::VarName),
                           make_tuple(string("y"), mY, NameFormat::Fixed),
                           make_tuple(string("yhat"), yhat, NameFormat::Fixed),
                           make_tuple(string("residual"), res, NameFormat::Fixed),
                           make_tuple(string("Stud_residual"), stu_res, NameFormat::Fixed),
                           make_tuple(string("SE"), betasSE, NameFormat::PrefixVarName),
                           make_tuple(string("TV"), betasTV, NameFormat::PrefixVarName),
                           make_tuple(string("localR2"), localR2, NameFormat::Fixed)});
    }
    else
    {
        createPredictionDistanceParameter();
        mBetas = regression(mX, mY);
        createResultLayer({make_tuple(string("%1"), mBetas, NameFormat::VarName)});
    }
}

mat CGwmRobustGWR::regression(const mat &x, const vec &y)
{
    return regressionSerial(x, y);
}

mat CGwmRobustGWR::regressionHatmatrix(const mat &x, const vec &y, mat &betasSE, vec &shat, vec &qdiag, mat &S)
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

mat CGwmRobustGWR::robustGWRCaliFirst(const mat &x, const vec &y, mat &betasSE, vec &shat, vec &qDiag, mat &S)
{
    mat betas = (this->*mRegressionHatmatrixFunction)(x, y, betasSE, shat, qDiag, S);
    //  ------------- 计算W.vect
    // vec yhat = fitted(x, betas);
    vec yhat = sum(betas % x, 1);
    vec residual = y - yhat;
    // 诊断信息
    GwmRegressionDiagnostic diagnostic;
    diagnostic = CalcDiagnostic(x, y, betas, shat);
    double trS = shat(0), trStS = shat(1);
    double nDp = x.n_rows;
    double sigmaHat = diagnostic.RSS / (nDp * 1.0 - 2 * trS + trStS);
    vec studentizedResidual = residual / sqrt(sigmaHat * qDiag);

    vec WVect(nDp, fill::zeros);

    //生成W.vect
    for (int i = 0; i < studentizedResidual.size(); i++)
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
    betas = (this->*mRegressionHatmatrixFunction)(x, y, betasSE, shat, qDiag, S);
    return betas;
}

mat CGwmRobustGWR::robustGWRCaliSecond(const mat &x, const vec &y, mat &betasSE, vec &shat, vec &qDiag, mat &S)
{
    int nDp = x.n_rows;
    double iter = 0;
    double diffmse = 1;
    double delta = 1.0e-5;
    double maxiter = 20;
    mat betas = (this->*mRegressionHatmatrixFunction)(x, y, betasSE, shat, qDiag, S);
    //计算residual
    // vec yHat = fitted(x, betas);
    vec yHat = sum(betas % x, 1);
    vec residual = y - yHat;
    //计算mse
    double mse = sum((residual % residual)) / residual.size();
    //计算WVect
    mWeightMask = filtWeight(residual, mse);
    while (diffmse > delta && iter < maxiter)
    {
        double oldmse = mse;
        betas = (this->*mRegressionHatmatrixFunction)(x, y, betasSE, shat, qDiag, S);
        //计算residual
        // yHat = fitted(x, betas);
        yHat = sum(betas % x, 1);
        residual = y - yHat;
        mse = sum((residual % residual)) / residual.size();
        mWeightMask = filtWeight(residual, mse);
        diffmse = abs(oldmse - mse) / mse;
        iter = iter + 1;
    }
    return betas;
}

mat CGwmRobustGWR::regressionHatmatrixSerial(const mat &x, const vec &y, mat &betasSE, vec &shat, vec &qDiag, mat &S)
{
    uword nDp = x.n_rows, nVar = x.n_cols;
    mat betas(nVar, nDp, fill::zeros);
    betasSE = mat(nVar, nDp, fill::zeros);
    shat = vec(2, fill::zeros);
    qDiag = vec(nDp, fill::zeros);
    S = mat(isStoreS() ? nDp : 1, nDp, fill::zeros);
    for (uword i = 0; i < nDp; i++)
    {
        vec w = mSpatialWeight.weightVector(mRegressionDistanceParameter, i) % mWeightMask;
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
            vec p = -si.t();
            p(i) += 1.0;
            qDiag += p % p;
            S.row(isStoreS() ? i : 0) = si;
        }
        catch (std::exception e)
        {
            throw e;
        }
    }
    betasSE = betasSE.t();
    return betas.t();
}

void CGwmRobustGWR::setParallelType(const ParallelType &type)
{
    if (type & parallelAbility())
    {
        mParallelType = type;
        switch (type)
        {
        case ParallelType::SerialOnly:
            mRegressionHatmatrixFunction = &CGwmRobustGWR::regressionHatmatrixSerial;
            break;
#ifdef ENABLE_OpenMP
        case ParallelType::OpenMP:
            mRegressionHatmatrixFunction = &CGwmRobustGWR::regressionHatmatrixOmp;
            break;
#endif
        default:
            mRegressionHatmatrixFunction = &CGwmRobustGWR::regressionHatmatrixSerial;
            break;
        }
    }
}

#ifdef ENABLE_OpenMP
mat CGwmRobustGWR::regressionHatmatrixOmp(const mat &x, const vec &y, mat &betasSE, vec &shat, vec &qDiag, mat &S)
{
    emit message("Regression ...");
    int nDp = x.n_rows, nVar = x.n_cols;
    mat betas(nVar, nDp, fill::zeros);
    betasSE = mat(nVar, nDp, fill::zeros);
    S = mat(isStoreS() ? nDp : 1, nDp, fill::zeros);
    mat shat_all(2, mOmpThreadNum, fill::zeros);
    mat qDiag_all(nDp, mOmpThreadNum, fill::zeros);
    int current = 0;
#pragma omp parallel for num_threads(mOmpThreadNum)
    for (int i = 0; i < nDp; i++)
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
            vec p = -si.t();
            p(i) += 1.0;
            qDiag_all.col(thread) += p % p;
            S.row(isStoreS() ? i : 0) = si;
        }
        catch (std::exception e)
        {
            throw e;
        }
    }
    shat = sum(shat_all, 1);
    qDiag = sum(qDiag_all, 1);
    betasSE = betasSE.t();
    return betas.t();
}
#endif

vec CGwmRobustGWR::filtWeight(vec residual, double mse)
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