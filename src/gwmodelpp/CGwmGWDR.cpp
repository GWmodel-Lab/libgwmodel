#include "CGwmGWDR.h"
#include <assert.h>
#include <exception>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_multimin.h>
#include <gsl/gsl_errno.h>
#include "GwmLogger.h"

#ifdef ENABLE_OPENMP
#include <omp.h>
#endif

using namespace std;

GwmRegressionDiagnostic CGwmGWDR::CalcDiagnostic(const mat& x, const vec& y, const mat& betas, const vec& shat)
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

mat CGwmGWDR::fit()
{
    uword nDims = mCoords.n_cols;
    
    // Set coordinates matrices.
    for (size_t m = 0; m < nDims; m++)
    {
        mSpatialWeights[m].distance()->makeParameter({ vec(mCoords.col(m)), vec(mCoords.col(m)) });
    }

    // Select Independent Variable
    if (mEnableIndepVarSelect)
    {
        vector<size_t> indep_vars;
        for (size_t i = (mHasIntercept ? 1 : 0); i < mX.n_cols; i++)
        {
            indep_vars.push_back(i);
        }
        CGwmVariableForwardSelector selector(indep_vars, mIndepVarSelectThreshold);
        mSelectedIndepVars = selector.optimize(this);
        if (mSelectedIndepVars.size() > 0)
        {
            mX = mX.cols(CGwmVariableForwardSelector::index2uvec(mSelectedIndepVars, mHasIntercept));
            mIndepVarCriterionList = selector.indepVarsCriterion();
        }
    }

    uword nDp = mCoords.n_rows, nVars = mX.n_cols;

    if (mEnableBandwidthOptimize)
    {
        for (auto&& sw : mSpatialWeights)
        {
            CGwmBandwidthWeight* bw = sw.weight<CGwmBandwidthWeight>();
            // Set Initial value
            double lower = bw->adaptive() ? nVars + 1 : sw.distance()->minDistance();
            double upper = bw->adaptive() ? nDp : sw.distance()->maxDistance();
            if (bw->bandwidth() <= lower || bw->bandwidth() >= upper)
            {
                bw->setBandwidth(upper * 0.618);
            }
        }
        vector<CGwmBandwidthWeight*> bws;
        for (auto&& iter : mSpatialWeights)
        {
            bws.push_back(iter.weight<CGwmBandwidthWeight>());
        }
        CGwmGWDRBandwidthOptimizer optimizer(bws);
        int status = optimizer.optimize(this, mCoords.n_rows, mBandwidthOptimizeMaxIter, mBandwidthOptimizeEps, mBandwidthOptimizeStep);
        if (status)
        {
            throw runtime_error("[CGwmGWDR::run] Bandwidth optimization invoke failed.");
        }
    }
    
    // Has hatmatrix
    mBetas = (this->*mFitFunction)(mX, mY, mBetasSE, mSHat, mQDiag, mS);
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
    vec localR2 = vec(nDp, arma::fill::zeros);
    for (uword i = 0; i < nDp; i++)
    {
        vec w(nDp, arma::fill::ones);
        for (auto&& sw : mSpatialWeights)
        {
            w = w % sw.weightVector(i);
        }
        double tss = sum(dybar2 % w);
        double rss = sum(dyhat2 % w);
        localR2(i) = (tss - rss) / tss;
    }
    
    return mBetas;
}

mat CGwmGWDR::predictSerial(const mat& locations, const mat& x, const vec& y)
{
    uword nDp = locations.n_rows, nVar = mX.n_cols;
    mat betas(nVar, nDp, arma::fill::zeros);
    for (size_t i = 0; i < nDp; i++)
    {
        vec w(nDp, arma::fill::ones);
        for (auto&& sw : mSpatialWeights)
        {
            vec w_m = sw.weightVector(i);
            w = w % w_m;
        }
        mat ws(1, nVar, arma::fill::ones);
        mat xtw = (x %(w * ws)).t();
        mat xtwx = xtw * x;
        mat xtwy = x.t() * (w % y);
        try
        {
            mat xtwx_inv = xtwx.i();
            betas.col(i) = xtwx_inv * xtwy;
        }
        catch(const std::exception& e)
        {
            GWM_LOG_ERROR(e.what());
            throw e;
        }
    }
    return betas.t();
}

#ifdef ENABLE_OPENMP
mat CGwmGWDR::predictOmp(const mat& locations, const mat& x, const vec& y)
{
    uword nDp = locations.n_rows, nVar = mX.n_cols;
    mat betas(nVar, nDp, arma::fill::zeros);
    bool success = true;
    std::exception except;
#pragma omp parallel for num_threads(mOmpThreadNum)
    for (int i = 0; (uword)i < nDp; i++)
    {
        if (success)
        {
            vec w(nDp, arma::fill::ones);
            for (auto&& sw : mSpatialWeights)
            {
                vec w_m = sw.weightVector(i);
                w = w % w_m;
            }
            mat ws(1, nVar, arma::fill::ones);
            mat xtw = (x %(w * ws)).t();
            mat xtwx = xtw * x;
            mat xtwy = x.t() * (w % y);
            try
            {
                mat xtwx_inv = xtwx.i();
                betas.col(i) = xtwx_inv * xtwy;
            }
            catch(const std::exception& e)
            {
                GWM_LOG_ERROR(e.what());
                except = e;
                success = false;
            }
        }
    }
    if (!success)
    {
        throw except;
    }
    return betas.t();
}
#endif

mat CGwmGWDR::fitSerial(const mat& x, const vec& y, mat& betasSE, vec& shat, vec& qdiag, mat& S)
{
    uword nDp = mCoords.n_rows, nVar = x.n_cols;
    mat betas(nVar, nDp, arma::fill::zeros);
    betasSE = mat(nVar, nDp, arma::fill::zeros);
    qdiag = vec(nDp, arma::fill::zeros);
    S = mat(isStoreS() ? nDp : 1, nDp, arma::fill::zeros);
    mat rowsumSE(nDp, 1, arma::fill::ones);
    vec s_hat1(nDp, arma::fill::zeros), s_hat2(nDp, arma::fill::zeros);
    for (size_t i = 0; i < nDp; i++)
    {
        vec w(nDp, arma::fill::ones);
        for (auto&& sw : mSpatialWeights)
        {
            vec w_m = sw.weightVector(i);
            w = w % w_m;
        }
        mat ws(1, nVar, arma::fill::ones);
        mat xtw = trans(x %(w * ws));
        mat xtwx = xtw * x;
        mat xtwy = trans(x) * (w % y);
        try
        {
            mat xtwx_inv = xtwx.i();
            betas.col(i) = xtwx_inv * xtwy;
            // hatmatrix
            mat ci = xtwx_inv * xtw;
            mat si = x.row(i) * ci;
            betasSE.col(i) = (ci % ci) * rowsumSE;
            s_hat1(i) = si(0, i);
            s_hat2(i) = det(si * si.t());
            mat onei(1, nDp, arma::fill::zeros);
            onei(i) = 1;
            mat p = (onei - si).t();
            qdiag += p % p;
            S.row(isStoreS() ? i : 0) = si;
        }
        catch(const std::exception& e)
        {
            GWM_LOG_ERROR(e.what());
            throw e;
        }
    }
    shat = {sum(s_hat1), sum(s_hat2)};
    betasSE = betasSE.t();
    return betas.t();
}

#ifdef ENABLE_OPENMP
mat CGwmGWDR::fitOmp(const mat& x, const vec& y, mat& betasSE, vec& shat, vec& qdiag, mat& S)
{
    uword nDp = mCoords.n_rows, nVar = x.n_cols;
    mat betas(nVar, nDp, arma::fill::zeros);
    betasSE = mat(nVar, nDp, arma::fill::zeros);
    qdiag = vec(nDp, arma::fill::zeros);
    S = mat(isStoreS() ? nDp : 1, nDp, arma::fill::zeros);
    mat rowsumSE(nDp, 1, arma::fill::ones);
    mat s_hat_all(2, mOmpThreadNum, arma::fill::zeros);
    mat qdiag_all(nDp, mOmpThreadNum, arma::fill::zeros);
    bool success = true;
    std::exception except;
#pragma omp parallel for num_threads(mOmpThreadNum)
    for (int i = 0; (uword)i < nDp; i++)
    {
        int thread = omp_get_thread_num();
        if (success)
        {
            vec w(nDp, arma::fill::ones);
            for (auto&& sw : mSpatialWeights)
            {
                vec w_m = sw.weightVector(i);
                w = w % w_m;
            }
            mat ws(1, nVar, arma::fill::ones);
            mat xtw = trans(x %(w * ws));
            mat xtwx = xtw * x;
            mat xtwy = trans(x) * (w % y);
            try
            {
                mat xtwx_inv = xtwx.i();
                betas.col(i) = xtwx_inv * xtwy;
                // hatmatrix
                mat ci = xtwx_inv * xtw;
                mat si = x.row(i) * ci;
                betasSE.col(i) = (ci % ci) * rowsumSE;
                s_hat_all(0, thread) += si(0, i);
                s_hat_all(1, thread) += det(si * si.t());
                mat onei(1, nDp, arma::fill::zeros);
                onei(i) = 1;
                mat p = (onei - si).t();
                qdiag_all.col(thread) += p % p;
                S.row(isStoreS() ? i : 0) = si;
            }
            catch(const std::exception& e)
            {
                GWM_LOG_ERROR(e.what());
                except = e;
            }
        }
    }
    if (!success)
    {
        throw except;
    }
    shat = sum(s_hat_all, 1);
    qdiag = sum(qdiag_all, 1);
    betasSE = betasSE.t();
    return betas.t();
}
#endif

double CGwmGWDR::bandwidthCriterionCVSerial(const vector<CGwmBandwidthWeight*>& bandwidths)
{
    uword nDp = mCoords.n_rows, nDim = mCoords.n_cols;
    double cv = 0.0;
    bool success = true;
    for (size_t i = 0; i < nDp; i++)
    {
        if (success)
        {
            vec w(nDp, arma::fill::ones);
            for (size_t m = 0; m < nDim; m++)
            {
                vec d_m = mSpatialWeights[m].distance()->distance(i);
                vec w_m = bandwidths[m]->weight(d_m);
                w = w % w_m;
            }
            w(i) = 0.0;
            mat xtw = (mX.each_col() % w).t();
            mat xtwx = xtw * mX;
            mat xtwy = xtw * mY;
            try
            {
                mat xtwx_inv = xtwx.i();
                vec beta = xtwx_inv * xtwy;
                double yhat = as_scalar(mX.row(i) * beta);
                double cv_i = mY(i) - yhat;
                cv += cv_i * cv_i;
            }
            catch(const std::exception& e)
            {
                GWM_LOG_ERROR(e.what());
                success = false;
            }
        }
    }
    return success ? cv : DBL_MAX;
}

#ifdef ENABLE_OPENMP
double CGwmGWDR::bandwidthCriterionCVOmp(const vector<CGwmBandwidthWeight*>& bandwidths)
{
    uword nDp = mCoords.n_rows, nDim = mCoords.n_cols;
    vec cv_all(mOmpThreadNum, arma::fill::zeros);
    bool success = true;
    std::exception except;
    for (size_t i = 0; i < nDp; i++)
    {
        int thread = omp_get_thread_num();
        if (success)
        {
            vec w(nDp, arma::fill::ones);
            for (size_t m = 0; m < nDim; m++)
            {
                vec d_m = mSpatialWeights[m].distance()->distance(i);
                vec w_m = bandwidths[m]->weight(d_m);
                w = w % w_m;
            }
            w(i) = 0.0;
            mat xtw = (mX.each_col() % w).t();
            mat xtwx = xtw * mX;
            mat xtwy = mX.t() * (w % mY);
            try
            {
                mat xtwx_inv = xtwx.i();
                vec beta = xtwx_inv * xtwy;
                double yhat = as_scalar(mX.row(i) * beta);
                double cv_i = abs(yhat - mY(i));
                cv_all(thread) += cv_i * cv_i;
            }
            catch(const std::exception& e)
            {
                GWM_LOG_ERROR(e.what());
                except = e;
                success = false;
            }
        }
    }
    if (!success)
    {
        throw except;
    }
    double cv = sum(cv_all);
    return success ? cv : DBL_MAX;
}
#endif

double CGwmGWDR::bandwidthCriterionAICSerial(const vector<CGwmBandwidthWeight*>& bandwidths)
{
    uword nDp = mCoords.n_rows, nDim = mCoords.n_cols, nVar = mX.n_cols;
    mat betas(nVar, nDp, fill::zeros);
    double trS = 0.0;
    bool flag = true;
    for (size_t i = 0; i < nDp; i++)
    {
        if (flag)
        {
            vec w(nDp, arma::fill::ones);
            for (size_t m = 0; m < nDim; m++)
            {
                vec d_m = mSpatialWeights[m].distance()->distance(i);
                vec w_m = bandwidths[m]->weight(d_m);
                w = w % w_m;
            }
            mat xtw = (mX.each_col() % w).t();
            mat xtwx = xtw * mX;
            mat xtwy = mX.t() * (w % mY);
            try
            {
                mat xtwx_inv = xtwx.i();
                betas.col(i) = xtwx_inv * xtwy;
                mat ci = xtwx_inv * xtw;
                mat si = mX.row(i) * ci;
                trS += si(0, i);
            }
            catch(const std::exception& e)
            {
                GWM_LOG_ERROR(e.what());
                flag = false;
            }
        }
    }
    if (!flag) return DBL_MAX;
    double value = CGwmGWDR::AICc(mX, mY, betas.t(), { trS, 0.0 });
    return isfinite(value) ? value : DBL_MAX;
}

#ifdef ENABLE_OPENMP
double CGwmGWDR::bandwidthCriterionAICOmp(const vector<CGwmBandwidthWeight*>& bandwidths)
{
    uword nDp = mCoords.n_rows, nDim = mCoords.n_cols, nVar = mX.n_cols;
    mat betas(nVar, nDp, arma::fill::zeros);
    vec trS_all(mOmpThreadNum, arma::fill::zeros);
    bool success = true;
    std::exception except;
#pragma omp parallel for num_threads(mOmpThreadNum)
    for (int i = 0; (uword)i < nDp; i++)
    {
        int thread = omp_get_thread_num();
        if (success)
        {
            vec w(nDp, arma::fill::ones);
            for (size_t m = 0; m < nDim; m++)
            {
                vec d_m = mSpatialWeights[m].distance()->distance(i);
                vec w_m = bandwidths[m]->weight(d_m);
                w = w % w_m;
            }
            mat xtw = (mX.each_col() % w).t();
            mat xtwx = xtw * mX;
            mat xtwy = mX.t() * (w % mY);
            try
            {
                mat xtwx_inv = xtwx.i();
                betas.col(i) = xtwx_inv * xtwy;
                mat ci = xtwx_inv * xtw;
                mat si = mX.row(i) * ci;
                trS_all(thread) += si(0, i);
            }
            catch(const std::exception& e)
            {
                GWM_LOG_ERROR(e.what());
                except = e;
                success = false;
            }
        }
    }
    if (!success)
    {
        return DBL_MAX;
    }
    double trS = sum(trS_all);
    double value = CGwmGWDR::AICc(mX, mY, betas.t(), { trS, 0.0 });
    return isfinite(value) ? value : DBL_MAX;
}
#endif

double CGwmGWDR::indepVarCriterionSerial(const vector<size_t>& indepVars)
{
    mat x = mX.cols(CGwmVariableForwardSelector::index2uvec(indepVars, mHasIntercept));
    vec y = mY;
    uword nDp = mCoords.n_rows, nVar = x.n_cols;
    mat betas(nVar, nDp, fill::zeros);
    double trS = 0.0;
    bool isGlobal = false, success = true;
    if (mEnableBandwidthOptimize) isGlobal = true;
    else
    {
        for (auto &&sw : mSpatialWeights)
        {
            if (sw.weight<CGwmBandwidthWeight>()->bandwidth() == 0.0) isGlobal = true;
        }
    }
    if (isGlobal)
    {
        mat xtwx = x.t() * x;
        vec xtwy = x.t() * y;
        try
        {
            mat xtwx_inv = inv_sympd(xtwx);
            vec beta = xtwx_inv * xtwy;
            betas = betas.each_col() + beta;
            mat ci = xtwx_inv * x.t();
            for (uword i = 0; i < nDp; i++)
            {
                mat si = x.row(i) * ci;
                trS += si(0, i);
            }
        }
        catch(const std::exception& e)
        {
            GWM_LOG_ERROR(e.what());
            success = false;
        }
    }
    else
    {
        for (uword i = 0; i < nDp; i++)
        {
            if (success)
            {
                vec w(nDp, arma::fill::ones);
                for (auto&& sw : mSpatialWeights)
                {
                    vec w_m = sw.weightVector(i);
                    w = w % w_m;
                }
                mat xtw = (x.each_col() % w).t();
                mat xtwx = xtw * x;
                mat xtwy = xtw * y;
                try
                {
                    mat xtwx_inv = inv_sympd(xtwx);
                    betas.col(i) = xtwx_inv * xtwy;
                    mat ci = xtwx_inv * xtw;
                    mat si = x.row(i) * ci;
                    trS += si(0, i);
                }
                catch (std::exception& e)
                {
                    GWM_LOG_ERROR(e.what());
                    success = false;
                }
            }
        }
    }
    double value = success ? CGwmGWDR::AICc(x, y, betas.t(), { trS, 0.0 }) : DBL_MAX;
    return isfinite(value) ? value : DBL_MAX;
}

#ifdef ENABLE_OPENMP
double CGwmGWDR::indepVarCriterionOmp(const vector<size_t>& indepVars)
{
    mat x = mX.cols(CGwmVariableForwardSelector::index2uvec(indepVars, mHasIntercept));
    vec y = mY;
    uword nDp = mCoords.n_rows, nVar = x.n_cols;
    mat betas(nVar, nDp, fill::zeros);
    double trS = 0.0;
    bool isGlobal = false, success = true;
    if (mEnableBandwidthOptimize) isGlobal = true;
    else
    {
        for (auto &&sw : mSpatialWeights)
        {
            if (sw.weight<CGwmBandwidthWeight>()->bandwidth() == 0.0) isGlobal = true;
        }
    }
    if (isGlobal)
    {
        mat xtwx = x.t() * x;
        vec xtwy = x.t() * y;
        try
        {
            mat xtwx_inv = inv_sympd(xtwx);
            vec beta = xtwx_inv * xtwy;
            betas = betas.each_col() + beta;
            mat ci = xtwx_inv * x.t();
            for (uword i = 0; i < nDp; i++)
            {
                mat si = x.row(i) * ci;
                trS += si(0, i);
            }
        }
        catch(const std::exception& e)
        {
            GWM_LOG_ERROR(e.what());
            success = false;
        }
        
    }
    else
    {
        vec trS_all(mOmpThreadNum, arma::fill::zeros);
#pragma omp parallel for num_threads(mOmpThreadNum)
        for (int i = 0; (uword)i < nDp; i++)
        {
            int thread = omp_get_thread_num();
            if (success)
            {
                vec w(nDp, arma::fill::ones);
                for (auto&& sw : mSpatialWeights)
                {
                    vec w_m = sw.weightVector(i);
                    w = w % w_m;
                }
                mat xtw = (x.each_col() % w).t();
                mat xtwx = xtw * x;
                mat xtwy = xtw * y;
                try
                {
                    mat xtwx_inv = inv_sympd(xtwx);
                    betas.col(i) = xtwx_inv * xtwy;
                    mat ci = xtwx_inv * xtw;
                    mat si = x.row(i) * ci;
                    trS_all(thread) += si(0, i);
                }
                catch (std::exception& e)
                {
                    GWM_LOG_ERROR(e.what());
                    success = false;
                }
            }
        }
        trS = sum(trS_all);
    }
    double value = success ? CGwmGWDR::AICc(x, y, betas.t(), { trS, 0.0 }) : DBL_MAX;
    return isfinite(value) ? value : DBL_MAX;
}
#endif

void CGwmGWDR::setBandwidthCriterionType(const BandwidthCriterionType& type)
{
    mBandwidthCriterionType = type;
    unordered_map<BandwidthCriterionType, BandwidthCriterionCalculator> mapper;
    switch (mParallelType)
    {
    case ParallelType::SerialOnly:
        mapper = {
            make_pair(BandwidthCriterionType::AIC, &CGwmGWDR::bandwidthCriterionAICSerial),
            make_pair(BandwidthCriterionType::CV, &CGwmGWDR::bandwidthCriterionCVSerial)
        };
        mBandwidthCriterionFunction = &CGwmGWDR::bandwidthCriterionAICSerial;
        break;
#ifdef ENABLE_OPENMP
    case ParallelType::OpenMP:
        mapper = {
            make_pair(BandwidthCriterionType::AIC, &CGwmGWDR::bandwidthCriterionAICOmp),
            make_pair(BandwidthCriterionType::CV, &CGwmGWDR::bandwidthCriterionCVOmp)
        };
        mBandwidthCriterionFunction = &CGwmGWDR::bandwidthCriterionAICOmp;
        break;
#endif
    default:
        mapper = {
            make_pair(BandwidthCriterionType::AIC, &CGwmGWDR::bandwidthCriterionAICSerial),
            make_pair(BandwidthCriterionType::CV, &CGwmGWDR::bandwidthCriterionCVSerial)
        };
        mBandwidthCriterionFunction = &CGwmGWDR::bandwidthCriterionAICSerial;
        break;
    }
    mBandwidthCriterionFunction = mapper[mBandwidthCriterionType];
}

void CGwmGWDR::setParallelType(const ParallelType& type)
{
    if (type & parallelAbility())
    {
        mParallelType = type;
        switch (type) {
        case ParallelType::SerialOnly:
            mPredictFunction = &CGwmGWDR::predictSerial;
            mFitFunction = &CGwmGWDR::fitSerial;
            mIndepVarCriterionFunction = &CGwmGWDR::indepVarCriterionSerial;
            break;
#ifdef ENABLE_OPENMP
        case ParallelType::OpenMP:
            mPredictFunction = &CGwmGWDR::predictOmp;
            mFitFunction = &CGwmGWDR::fitOmp;
            mIndepVarCriterionFunction= &CGwmGWDR::indepVarCriterionOmp;
            break;
#endif
        default:
            mPredictFunction = &CGwmGWDR::predictSerial;
            mFitFunction = &CGwmGWDR::fitSerial;
            mIndepVarCriterionFunction = &CGwmGWDR::indepVarCriterionSerial;
            break;
        }
    }
    setBandwidthCriterionType(mBandwidthCriterionType);
}

bool CGwmGWDR::isValid()
{
    if (CGwmSpatialAlgorithm::isValid())
    {
        if (!(mSpatialWeights.size() == mCoords.n_cols))
        {
            return false;
        }

        return true;
    }
    else return false;
}

double CGwmGWDRBandwidthOptimizer::criterion_function(const gsl_vector* bws, void* params)
{
    Parameter* p = static_cast<Parameter*>(params);
    CGwmGWDR* instance = p->instance;
    const vector<CGwmBandwidthWeight*>& bandwidths = *(p->bandwidths);
    double nFeature = double(p->featureCount);
    for (size_t m = 0; m < bandwidths.size(); m++)
    {
        double pbw = abs(gsl_vector_get(bws, m));
        bandwidths[m]->setBandwidth(pbw * nFeature);
    }
    return instance->bandwidthCriterion(bandwidths);
}

const int CGwmGWDRBandwidthOptimizer::optimize(CGwmGWDR* instance, uword featureCount, size_t maxIter, double eps, double step)
{
    size_t nDim = mBandwidths.size();
    gsl_multimin_fminimizer* minimizer = gsl_multimin_fminimizer_alloc(gsl_multimin_fminimizer_nmsimplex2rand, nDim);
    gsl_vector* targets = gsl_vector_alloc(nDim);
    gsl_vector* steps = gsl_vector_alloc(nDim);
    for (size_t m = 0; m < nDim; m++)
    {
        double target_value = mBandwidths[m]->adaptive() ? mBandwidths[m]->bandwidth() / double(featureCount) : mBandwidths[m]->bandwidth();
        gsl_vector_set(targets, m, target_value);
        gsl_vector_set(steps, m, step);
    }
    Parameter params = { instance, &mBandwidths, featureCount };
    gsl_multimin_function function = { criterion_function, nDim, &params };
    int status = gsl_multimin_fminimizer_set(minimizer, &function, targets, steps);
    if (status == GSL_SUCCESS)
    {
        size_t iter = 0;
        double size = DBL_MAX;
        do
        {
            iter++;
            status = gsl_multimin_fminimizer_iterate(minimizer);
            if (status)
                break;
            size = gsl_multimin_fminimizer_size(minimizer);
            status = gsl_multimin_test_size(size, eps);
            #ifdef _DEBUG
            for (size_t m = 0; m < nDim; m++)
            {
                cout << gsl_vector_get(minimizer->x, m) << ",";
            }
            cout << minimizer->fval << ",";
            cout << size << "\n";
            #endif
        } 
        while (status == GSL_CONTINUE && iter < maxIter);
        #ifdef _DEBUG
        for (size_t m = 0; m < nDim; m++)
        {
            cout << gsl_vector_get(minimizer->x, m) << ",";
        }
        cout << minimizer->fval << ",";
        cout << size << "\n";
        #endif
        for (size_t m = 0; m < nDim; m++)
        {
            double pbw = abs(gsl_vector_get(minimizer->x, m));
            // pbw = (pbw > 1.0 ? 1.0 : pbw);
            mBandwidths[m]->setBandwidth(round(pbw * featureCount));
        }
    }
    gsl_multimin_fminimizer_free(minimizer);
    gsl_vector_free(targets);
    gsl_vector_free(steps);
    return status;
}
