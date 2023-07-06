#include "GWRGeneralized.h"
#include <exception>
#include "BandwidthSelector.h"
#include "VariableForwardSelector.h"
#include <assert.h>
#include "math.h"
#include "gsl/gsl_sf.h"
#include "gsl/gsl_randist.h"
#include "Logger.h"
#include "GeneralizedLinearModel.h"

#ifdef ENABLE_OPENMP
#include <omp.h>
#endif

using namespace arma;
using namespace std;
using namespace gwm;

mat GWRGeneralized::fit()
{
    GWM_LOG_STAGE("Initializing");
    // 初始化
    // setXY(mX, mY, mSourceLayer, mDepVar, mIndepVars);
    uword nVar = mX.n_cols;
    uword nDp = mCoords.n_rows, nRp = mHasRegressionData ? mRegressionData.n_rows : nDp;
    // 点位初始化
    createDistanceParameter();
    GWM_LOG_STOP_RETURN(mStatus, mat(nDp, nVar, arma::fill::zeros));

    // 优选带宽
    if (mIsAutoselectBandwidth)
    {
        GWM_LOG_STAGE("Bandwidth selection");
        // emit message(string("Automatically selecting bandwidth ..."));
        // emit tick(0, 0);
        BandwidthWeight *bw0 = mSpatialWeight.weight<BandwidthWeight>();
        double lower = bw0->adaptive() ? 20 : 0.0;
        double upper = bw0->adaptive() ? nDp : mSpatialWeight.distance()->maxDistance();
        
        GWM_LOG_INFO(IBandwidthSelectable::infoBandwidthCriterion(bw0).str());
        BandwidthSelector selector(bw0, lower, upper);
        BandwidthWeight *bw = selector.optimize(this);
        if (bw)
        {
            mSpatialWeight.setWeight(bw);
            mBandwidthSelectionCriterionList = selector.bandwidthCriterion();
        }
        GWM_LOG_STOP_RETURN(mStatus, mat(nDp, nVar, arma::fill::zeros));
    }

    GWM_LOG_STAGE("Preparing");
    mBetas = mat(nVar, nRp, fill::zeros);
    if (mHasHatMatrix)
    {
        mBetasSE = mat(nVar, nDp, fill::zeros);
        mShat = vec(2, fill::zeros);
    }
    mWtMat1 = mat(nDp, nDp, fill::zeros);
    mWtMat2 = mat(nRp, nDp, fill::zeros);
    if (mHasRegressionData)
    {
        for (uword i = 0; i < nRp; i++)
        {
            vec weight = mSpatialWeight.weightVector(i);
            mWtMat2.col(i) = weight;
        }
        for (uword i = 0; i < nDp; i++)
        {
            vec weight = mSpatialWeight.weightVector(i);
            mWtMat1.col(i) = weight;
        }
    }
    else
    {
        for (uword i = 0; i < nDp; i++)
        {
            vec weight = mSpatialWeight.weightVector(i);
            mWtMat2.col(i) = weight;
        }
        mWtMat1 = mWtMat2;
    }

    //bool isAllCorrect = true;
    GWM_LOG_STAGE("Calibrating GLM model");
    CalGLMModel(mX, mY);

    GWM_LOG_STAGE("Model fitting");
    mBetas = (this->*mGGWRfitFunction)(mX, mY);
    GWM_LOG_STOP_RETURN(mStatus, mat(nDp, nVar, arma::fill::zeros));

    GWM_LOG_STAGE("Model Diagnostic");
    if (mHasHatMatrix)
    {
        if (mFamily == Family::Poisson)
        {
            mat betasTV = mBetas / mBetasSE;
            mBetas = trans(mBetas);
            mBetasSE = trans(mBetasSE);
            betasTV = trans(betasTV);
            double trS = mShat(0);
            //double trStS = mShat(1);

            mat yhat = exp(Fitted(mX, mBetas));
            mat res = mY - yhat;

            // 计算诊断信息
            double AIC = mGwDev + 2 * trS;
            double AICc = AIC + 2 * trS * (trS + 1) / (nDp - trS - 1);
            double R2 = 1 - mGwDev / (mGLMDiagnostic.NullDev); // pseudo.R2 <- 1 - gw.dev/null.dev
            vec vDiags(4);
            vDiags(0) = AIC;
            vDiags(1) = AICc;
            vDiags(2) = mGwDev;
            vDiags(3) = R2;
            mDiagnostic = GWRGeneralizedDiagnostic(vDiags);

            return mBetas;
        }
        else
        {
            mat n = vec(mY.n_rows, fill::ones);
            mBetas = trans(mBetas);

            double trS = mShat(0);
            //double trStS = mShat(1);

            vec yhat = Fitted(mX, mBetas);
            yhat = exp(yhat) / (1 + exp(yhat));

            vec res = mY - yhat;
            vec Dev = log(1 / ((mY - n + yhat) % (mY - n + yhat)));
            double gwDev = sum(Dev);
            vec residual2 = res % res;
            //double rss = sum(residual2);
            for (uword i = 0; i < nDp; i++)
            {
                mBetasSE.col(i) = sqrt(mBetasSE.col(i));
                //            mBetasTV.col(i) = mBetas.col(i) / mBetasSE.col(i);
            }
            mBetasSE = trans(mBetasSE);
            mat betasTV = mBetas / mBetasSE;

            double AIC = gwDev + 2 * trS;
            double AICc = AIC + 2 * trS * (trS + 1) / (nDp - trS - 1);
            double R2 = 1 - gwDev / (mGLMDiagnostic.NullDev); // pseudo.R2 <- 1 - gw.dev/null.dev
            vec vDiags(4);
            vDiags(0) = AIC;
            vDiags(1) = AICc;
            vDiags(2) = gwDev;
            vDiags(3) = R2;
            mDiagnostic = GWRGeneralizedDiagnostic(vDiags);

            return mBetas;
        }
    }
    else
    {
        mBetas = trans(mBetas);
        return mBetas;
    }
}

void GWRGeneralized::createPredictionDistanceParameter(const arma::mat &locations)
{
    if (mSpatialWeight.distance()->type() == Distance::DistanceType::CRSDistance ||
        mSpatialWeight.distance()->type() == Distance::DistanceType::MinkwoskiDistance)
    {
        mSpatialWeight.distance()->makeParameter({locations, mCoords});
    }
}
mat GWRGeneralized::predict(const mat& locations)
{
    uword nDp = mCoords.n_rows, nVars = mX.n_cols;
    mHasHatMatrix = false;
    createPredictionDistanceParameter(locations);
    GWM_LOG_STOP_RETURN(mStatus, mat(nDp, nVars, arma::fill::zeros));

    mBetas = (this->*mGGWRfitFunction)(mX, mY);
    GWM_LOG_STOP_RETURN(mStatus, mat(nDp, nVars, arma::fill::zeros));

    return mBetas;
}

void GWRGeneralized::CalGLMModel(const mat &x, const vec &y)
{
    uword nDp = mCoords.n_rows, nVar = x.n_cols;
    GeneralizedLinearModel mGlm;
    mGlm.setX(x);
    mGlm.setY(y);
    mGlm.setFamily(mFamily);
    mGlm.fit();
    double nulldev = mGlm.nullDev();
    double dev = mGlm.dev();
    double pseudor2 = 1 - dev / nulldev;
    double aic = dev + 2 * nVar;
    double aicc = aic + 2 * nVar * (nVar + 1) / (nDp - nVar - 1);
    vec vGLMDiags(5);
    vGLMDiags(0) = aic;
    vGLMDiags(1) = aicc;
    vGLMDiags(2) = nulldev;
    vGLMDiags(3) = dev;
    vGLMDiags(4) = pseudor2;
    mGLMDiagnostic = GLMDiagnostic(vGLMDiags);
}

/* mat GWRGeneralized::fit(const mat &x, const vec &y, mat &betasSE, vec &shat, vec &qdiag, mat &S){
    return (this->*mGGWRfitFunction)(x, y);
} */
mat GWRGeneralized::fitPoissonSerial(const mat &x, const vec &y)
{
    uword nDp = mCoords.n_rows, nRp = mHasRegressionData ? mRegressionData.n_rows : nDp;
    uword nVar = x.n_cols;
    mat betas = mat(nVar, nRp, fill::zeros);

    vec mu = (this->*mCalWtFunction)(x, y, mWtMat1);
    mGwDev = 0.0;
    for (uword i = 0; i < nDp; i++)
    {
        if (y[i] != 0)
        {
            mGwDev = mGwDev + 2 * (y[i] * (log(y[i] / mu[i]) - 1) + mu[i]);
        }
        else
        {
            mGwDev = mGwDev + 2 * mu[i];
        }
    }
    // emit message(tr("Calibrating GGWR model..."));
    // emit tick(0, nDp);
    bool isAllCorrect = true;
    exception except;
    bool isStoreS = (nDp <= 8192);
    mat ci, s_ri, S(isStoreS ? nDp : 1, nDp, fill::zeros);
    if (mHasHatMatrix)
    {
        for (uword i = 0; i < nDp; i++)
        {
            GWM_LOG_STOP_BREAK(mStatus);
            try
            {
                vec wi = mWtMat2.col(i);
                vec gwsi = gwFit(x, myAdj, wi % mWt2, i, ci, s_ri);
                betas.col(i) = gwsi;
                mat invwt2 = 1.0 / mWt2;
                S.row(isStoreS ? i : 0) = s_ri;
                mat temp = mat(ci.n_rows, ci.n_cols);
                for (uword j = 0; j < ci.n_rows; j++)
                {
                    temp.row(j) = ci.row(j) % trans(invwt2);
                }
                mBetasSE.col(i) = diag(temp * trans(ci));

                mShat(0) += s_ri(0, i);
                mShat(1) += det(s_ri * trans(s_ri));

                mBetasSE.col(i) = sqrt(mBetasSE.col(i));
            }
            catch (const exception& e)
            {
                GWM_LOG_ERROR(e.what());
                except=e;
                isAllCorrect = false;
                
            }
            GWM_LOG_PROGRESS(i + 1, nDp);
        }
    }
    else
    {
        for (uword i = 0; i < nRp; i++)
        {
            GWM_LOG_STOP_BREAK(mStatus);
            try
            {
                vec wi = mWtMat2.col(i);
                vec gwsi = gwPredict(x, myAdj, wi * mWt2);
                betas.col(i) = gwsi;
            }
            catch (const exception& e)
            {
                GWM_LOG_ERROR(e.what());
                except=e;
                isAllCorrect = false;
            }
            GWM_LOG_PROGRESS(i + 1, nDp);
        }
    }
    if (!isAllCorrect)
    {
        throw except;
    }
    return betas;
}

#ifdef ENABLE_OPENMP
mat GWRGeneralized::fitPoissonOmp(const mat &x, const vec &y)
{
    uword nDp = mCoords.n_rows, nRp = mHasRegressionData ? mRegressionData.n_rows : nDp;
    uword nVar = x.n_cols;
    mat betas = mat(nVar, nRp, fill::zeros);

    vec mu = (this->*mCalWtFunction)(x, y, mWtMat1);

    mGwDev = 0.0;
    for (uword i = 0; i < nDp; i++)
    {
        if (y[i] != 0)
        {
            mGwDev = mGwDev + 2 * (y[i] * (log(y[i] / mu[i]) - 1) + mu[i]);
        }
        else
        {
            mGwDev = mGwDev + 2 * mu[i];
        }
    }
    // emit message(tr("Calibrating GGWR model..."));
    // emit tick(0, nDp);
    bool isAllCorrect = true;
    exception except;
    bool isStoreS = (nDp <= 8192);
    mat S(isStoreS ? nDp : 1, nDp, fill::zeros);
    //int current = 0;
    if (mHasHatMatrix)
    {
        mat shat = mat(2, mOmpThreadNum, fill::zeros);
#pragma omp parallel for num_threads(mOmpThreadNum)
        for (int i = 0; i < (int)nDp; i++)
        {
            GWM_LOG_STOP_CONTINUE(mStatus);
            mat ci, s_ri;
            if (true)
            {
                try
                {
                    int thread = omp_get_thread_num();
                    vec wi = mWtMat2.col(i);
                    vec gwsi = gwFit(x, myAdj, wi % mWt2, i, ci, s_ri);
                    betas.col(i) = gwsi;
                    mat invwt2 = 1.0 / mWt2;
                    S.row(isStoreS ? i : 0) = s_ri;
                    mat temp = mat(ci.n_rows, ci.n_cols);
                    for (int j = 0; (uword)j < ci.n_rows; j++)
                    {
                        temp.row(j) = ci.row(j) % trans(invwt2);
                    }
                    mBetasSE.col(i) = diag(temp * trans(ci));

                    shat(0, thread) += s_ri(0, i);
                    shat(1, thread) += det(s_ri * trans(s_ri));

                    mBetasSE.col(i) = sqrt(mBetasSE.col(i));
                    //            mBetasTV.col(i) = mBetas.col(i) / mBetasSE.col(i);

                    // emit tick(current++, nDp);
                }
                catch (const exception& e)
                {
                    GWM_LOG_ERROR(e.what());
                    except=e;
                    isAllCorrect = false;
                }
            }
            GWM_LOG_PROGRESS(i + 1, nDp);
        }
        mShat(0) = sum(trans(shat.row(0)));
        mShat(1) = sum(trans(shat.row(1)));
    }
    else
    {
#pragma omp parallel for num_threads(mOmpThreadNum)
        for (int i = 0; i < (int)nRp; i++)
        {
            GWM_LOG_STOP_CONTINUE(mStatus);
            if (true)
            {
                try
                {
                    vec wi = mWtMat2.col(i);
                    vec gwsi = gwPredict(x, myAdj, wi * mWt2);
                    betas.col(i) = gwsi;
                    // emit tick(current++, nRp);
                }
                catch (const exception& e)
                {
                    GWM_LOG_ERROR(e.what());
                    except=e;
                    isAllCorrect = false;
                    // emit error(e.what());
                }
            }
            GWM_LOG_PROGRESS(i + 1, nDp);
        }
    }
    if(!isAllCorrect){
        throw except;
    }
    return betas;
}
#endif

#ifdef ENABLE_OPENMP
mat GWRGeneralized::fitBinomialOmp(const mat &x, const vec &y)
{
    uword nVar = x.n_cols;
    uword nDp = mCoords.n_rows, nRp = mHasRegressionData ? mRegressionData.n_rows : nDp;
    //    mat S = mat(nDp,nDp);
    //    mat n = vec(mY.n_rows,fill::ones);
    mat betas = mat(nVar, nRp, fill::zeros);

    vec mu = (this->*mCalWtFunction)(x, y, mWtMat1);
    // emit message(tr("Calibrating GGWR model..."));
    // emit tick(0, nDp);
    bool isAllCorrect = true;
    exception except;
    bool isStoreS = (nDp <= 8192);
    mat S(isStoreS ? nDp : 1, nDp, fill::zeros);
    //    mat S = mat(uword(0), uword(0));
    //int current = 0;
    if (mHasHatMatrix)
    {
        mat shat = mat(mOmpThreadNum, 2, fill::zeros);
#pragma omp parallel for num_threads(mOmpThreadNum)
        for (int i = 0; i < (int)nDp; i++)
        {
            GWM_LOG_STOP_CONTINUE(mStatus);
            mat ci, s_ri;
            if (true)
            {
                try
                {
                    int thread = omp_get_thread_num();
                    vec wi = mWtMat1.col(i);
                    vec gwsi = gwFit(x, myAdj, wi % mWt2, i, ci, s_ri);
                    betas.col(i) = gwsi;
                    mat invwt2 = 1.0 / mWt2;
                    S.row(isStoreS ? i : 0) = s_ri;
                    mat temp = mat(ci.n_rows, ci.n_cols);
                    for (int j = 0; (uword)j < ci.n_rows; j++)
                    {
                        temp.row(j) = ci.row(j) % trans(invwt2);
                    }
                    mBetasSE.col(i) = diag(temp * trans(ci));
                    shat(thread, 0) += s_ri(0, i);
                    shat(thread, 1) += det(s_ri * trans(s_ri));
                    // emit tick(current++, nDp);
                }
                catch (const exception& e)
                {
                    GWM_LOG_ERROR(e.what());
                    except=e;
                    isAllCorrect = false;
                    // emit error(e.what());
                }
            }
            GWM_LOG_PROGRESS(i + 1, nDp);
        }
        mShat(0) = sum(shat.col(0));
        mShat(1) = sum(shat.col(1));
    }
    else
    {
#pragma omp parallel for num_threads(mOmpThreadNum)
        for (int i = 0; i < (int)nRp; i++)
        {
            GWM_LOG_STOP_CONTINUE(mStatus);
            if (true)
            {
                try
                {
                    vec wi = mWtMat2.col(i);
                    vec gwsi = gwPredict(x, myAdj, wi * mWt2);
                    mBetas.col(i) = gwsi;
                    // emit tick(current++, nRp);
                }
                catch (const exception& e)
                {
                    GWM_LOG_ERROR(e.what());
                    except=e;
                    isAllCorrect = false;
                    // emit error(e.what());
                }
            }
            GWM_LOG_PROGRESS(i + 1, nDp)
        }
    }
    if(!isAllCorrect){
        throw except;
    }
    return betas;
}
#endif

mat GWRGeneralized::fitBinomialSerial(const mat &x, const vec &y)
{
    uword nDp = mCoords.n_rows, nRp = mHasRegressionData ? mRegressionData.n_rows : nDp;
    uword nVar = x.n_cols;
    //    mat S = mat(nDp,nDp);
    //    mat n = vec(mY.n_rows,fill::ones);
    mat betas = mat(nVar, nRp, fill::zeros);

    vec mu = (this->*mCalWtFunction)(x, y, mWtMat1);

    bool isAllCorrect = true;
    exception except;
    bool isStoreS = (nDp <= 8192);
    mat ci;
    mat s_ri, S(isStoreS ? nDp : 1, nDp, fill::zeros);
    //    mat S = mat(uword(0), uword(0));
    if (mHasHatMatrix)
    {
        for (uword i = 0; i < nDp; i++)
        {
            GWM_LOG_STOP_BREAK(mStatus);
            try
            {
                vec wi = mWtMat1.col(i);
                vec gwsi = gwFit(x, myAdj, wi % mWt2, i, ci, s_ri);
                betas.col(i) = gwsi;
                mat invwt2 = 1.0 / mWt2;
                S.row(isStoreS ? i : 0) = s_ri;
                mat temp = mat(ci.n_rows, ci.n_cols);
                for (int j = 0; (uword)j < ci.n_rows; j++)
                {
                    temp.row(j) = ci.row(j) % trans(invwt2);
                }
                mBetasSE.col(i) = diag(temp * trans(ci));
                mShat(0) += s_ri(0, i);
                mShat(1) += det(s_ri * trans(s_ri));
            }
            catch (const exception& e)
            {
                GWM_LOG_ERROR(e.what());
                except=e;
                isAllCorrect = false;
            }
            GWM_LOG_PROGRESS(i + 1, nDp);
        }
    }
    else
    {
        for (uword i = 0; i < nRp; i++)
        {
            GWM_LOG_STOP_BREAK(mStatus);
            try
            {
                vec wi = mWtMat2.col(i);
                vec gwsi = gwPredict(x, myAdj, wi * mWt2);
                mBetas.col(i) = gwsi;
                // emit tick(i, nRp);
            }
            catch (const exception& e)
            {
                GWM_LOG_ERROR(e.what());
                except=e;
                isAllCorrect = false;
                // emit error(e.what());
            }
            GWM_LOG_PROGRESS(i + 1, nRp);
        }
    }
    if(!isAllCorrect){
        throw except;
    }
    return betas;
}

double GWRGeneralized::bandwidthSizeGGWRCriterionCVSerial(BandwidthWeight *bandwidthWeight)
{
    uword n = mCoords.n_rows;
    vec cv = vec(n);
    mat wt = mat(n, n);
    for (uword i = 0; i < n; i++)
    {
        GWM_LOG_STOP_BREAK(mStatus);
        vec d = mSpatialWeight.distance()->distance(i);
        vec w = bandwidthWeight->weight(d);
        w.row(i) = 0;
        wt.col(i) = w;
    }
    GWM_LOG_STOP_RETURN(mStatus, DBL_MAX);
    
    (this->*mCalWtFunction)(mX, mY, wt);
    GWM_LOG_STOP_RETURN(mStatus, DBL_MAX);

    for (uword i = 0; i < n; i++)
    {
        GWM_LOG_STOP_BREAK(mStatus);
        mat wi = wt.col(i) % mWt2;
        vec gwsi = gwPredict(mX, myAdj, wi);
        mat yhatnoi = mX.row(i) * gwsi;
        if (mFamily == GWRGeneralized::Family::Poisson)
        {
            cv.row(i) = mY.row(i) - exp(yhatnoi);
        }
        else
        {
            cv.row(i) = mY.row(i) - exp(yhatnoi) / (1 + exp(yhatnoi));
        }
    }
    GWM_LOG_STOP_RETURN(mStatus, DBL_MAX);

    vec cvsquare = trans(cv) * cv;
    double res = sum(cvsquare);
    if (mStatus == Status::Success && isfinite(res))
    {
        GWM_LOG_PROGRESS_PERCENT(exp(- abs(mBandwidthLastCriterion - res)));
        mBandwidthLastCriterion = res;
        return res;
    }
    else return DBL_MAX;
}

#ifdef ENABLE_OPENMP
double GWRGeneralized::bandwidthSizeGGWRCriterionCVOmp(BandwidthWeight *bandwidthWeight)
{
    uword n = mCoords.n_rows;
    vec cv = vec(n);
    mat wt = mat(n, n);
    //int current1 = 0, current2 = 0;
#pragma omp parallel for num_threads(mOmpThreadNum)
    for (int i = 0; i < (int)n; i++)
    {
        GWM_LOG_STOP_CONTINUE(mStatus);
        if (true)
        {
            vec d = mSpatialWeight.distance()->distance(i);
            vec w = bandwidthWeight->weight(d);
            w.row(i) = 0;
            wt.col(i) = w;
        }
    }
    GWM_LOG_STOP_RETURN(mStatus, DBL_MAX);

    (this->*mCalWtFunction)(mX, mY, wt);
    GWM_LOG_STOP_RETURN(mStatus, DBL_MAX);

#pragma omp parallel for num_threads(mOmpThreadNum)
    for (int i = 0; i < (int)n; i++)
    {
        GWM_LOG_STOP_CONTINUE(mStatus);
        if (true)
        {
            mat wi = wt.col(i) % mWt2;
            vec gwsi = gwPredict(mX, myAdj, wi);
            mat yhatnoi = mX.row(i) * gwsi;
            if (mFamily == GWRGeneralized::Family::Poisson)
            {
                cv.row(i) = mY.row(i) - exp(yhatnoi);
            }
            else
            {
                cv.row(i) = mY.row(i) - exp(yhatnoi) / (1 + exp(yhatnoi));
            }
        }
    }
    GWM_LOG_STOP_RETURN(mStatus, DBL_MAX);

    vec cvsquare = trans(cv) * cv;
    double res = sum(cvsquare);
    if (mStatus == Status::Success && isfinite(res))
    {
        GWM_LOG_PROGRESS_PERCENT(exp(- abs(mBandwidthLastCriterion - res)));
        mBandwidthLastCriterion = res;
        return res;
    }
    else return DBL_MAX;
}
#endif

double GWRGeneralized::bandwidthSizeGGWRCriterionAICSerial(BandwidthWeight *bandwidthWeight)
{
    uword n = mCoords.n_rows;
    vec cv = vec(n);
    mat S = mat(n, n);
    mat wt = mat(n, n);
    for (uword i = 0; i < n; i++)
    {
        GWM_LOG_STOP_BREAK(mStatus);
        vec d = mSpatialWeight.distance()->distance(i);
        vec w = bandwidthWeight->weight(d);
        wt.col(i) = w;
    }
    GWM_LOG_STOP_RETURN(mStatus, DBL_MAX);

    (this->*mCalWtFunction)(mX, mY, wt);
    GWM_LOG_STOP_RETURN(mStatus, DBL_MAX);

    vec trS = vec(1, fill::zeros);
    for (uword i = 0; i < n; i++)
    {
        GWM_LOG_STOP_BREAK(mStatus);
        vec wi = wt.col(i) % mWt2;
        mat Ci = CiMat(mX, wi);
        S.row(i) = mX.row(i) * Ci;
        trS(0) += S(i, i);
    }
    GWM_LOG_STOP_RETURN(mStatus, DBL_MAX);

    if (mStatus == Status::Success && S.is_finite())
    {
        double trs = double(trS(0));
        double AICc = -2 * mLLik + 2 * trs + 2 * trs * (trs + 1) / (n - trs - 1);
        GWM_LOG_PROGRESS_PERCENT(exp(- abs(mBandwidthLastCriterion - AICc)));
        mBandwidthLastCriterion = AICc;
        return AICc;
    }
    else return DBL_MAX;
}

#ifdef ENABLE_OPENMP
double GWRGeneralized::bandwidthSizeGGWRCriterionAICOmp(BandwidthWeight *bandwidthWeight)
{
    uword n = mCoords.n_rows;
    vec cv = vec(n);
    mat S = mat(n, n);
    mat wt = mat(n, n);
    //int current1 = 0, current2 = 0;
#pragma omp parallel for num_threads(mOmpThreadNum)
    for (int i = 0; i < (int)n; i++)
    {
        GWM_LOG_STOP_CONTINUE(mStatus);
        if (true)
        {
            vec d = mSpatialWeight.distance()->distance(i);
            vec w = bandwidthWeight->weight(d);
            wt.col(i) = w;
        }
    }
    GWM_LOG_STOP_RETURN(mStatus, DBL_MAX);
    
    (this->*mCalWtFunction)(mX, mY, wt);
    GWM_LOG_STOP_RETURN(mStatus, DBL_MAX);

    vec trS = vec(mOmpThreadNum, fill::zeros);
#pragma omp parallel for num_threads(mOmpThreadNum)
    for (int i = 0; i < (int)n; i++)
    {
        GWM_LOG_STOP_CONTINUE(mStatus);
        if (true)
        {
            int thread = omp_get_thread_num();
            vec wi = wt.col(i) % mWt2;
            mat Ci = CiMat(mX, wi);
            S.row(i) = mX.row(i) * Ci;
            trS(thread) += S(i, i);
        }
    }
    GWM_LOG_STOP_RETURN(mStatus, DBL_MAX);

    if (mStatus == Status::Success && S.is_finite())
    {
        double trs = double(trS(0));
        double AICc = -2 * mLLik + 2 * trs + 2 * trs * (trs + 1) / (n - trs - 1);
        GWM_LOG_PROGRESS_PERCENT(exp(- abs(mBandwidthLastCriterion - AICc)));
        mBandwidthLastCriterion = AICc;
        return AICc;
    }
    else return DBL_MAX;
}
#endif

vec GWRGeneralized::PoissonWtSerial(const mat &x, const vec &y, mat wt)
{
    uword varn = x.n_cols;
    uword dpn = x.n_rows;
    mat betas = mat(varn, dpn, fill::zeros);
    mat S = mat(dpn, dpn);
    uword itCount = 0;
    double oldLLik = 0.0;
    vec mu = y + 0.1;
    vec nu = log(mu);
    vec cv = vec(dpn);
    mWt2 = ones(dpn);
    mLLik = 0;

    while (itCount < mMaxiter)
    {
        myAdj = nu + (y - mu) / mu;
        for (uword i = 0; i < dpn; i++)
        {
            vec wi = wt.col(i);
            vec gwsi = gwPredict(x, myAdj, wi % mWt2);
            betas.col(i) = gwsi;
        }
        mat betas1 = trans(betas);
        nu = Fitted(x, betas1);
        mu = exp(nu);
        oldLLik = mLLik;
        vec lliktemp = dpois(y, mu);
        mLLik = sum(lliktemp);
        double diff = abs((oldLLik - mLLik) / mLLik);
        if (diff < mTol)
        {
            GWM_LOG_PROGRESS_PERCENT(exp(- (diff - mTol)));
            break;
        }
        mWt2 = mu;
        itCount++;
    }
    return mu;
}

#ifdef ENABLE_OPENMP
vec GWRGeneralized::PoissonWtOmp(const mat &x, const vec &y, mat wt)
{
    uword varn = x.n_cols;
    uword dpn = x.n_rows;
    mat betas = mat(varn, dpn, fill::zeros);
    mat S = mat(dpn, dpn);
    size_t itCount = 0;
    double oldLLik = 0.0;
    vec mu = y + 0.1;
    vec nu = log(mu);
    vec cv = vec(dpn);
    mWt2 = ones(dpn);
    mLLik = 0;
    while (itCount < mMaxiter)
    {
        myAdj = nu + (y - mu) / mu;
        for (uword i = 0; i < dpn; i++)
        {
            vec wi = wt.col(i);
            vec gwsi = gwPredict(x, myAdj, wi % mWt2);
            betas.col(i) = gwsi;
        }
        mat betas1 = trans(betas);
        nu = Fitted(x, betas1);
        mu = exp(nu);
        oldLLik = mLLik;
        vec lliktemp = dpois(y, mu);
        mLLik = sum(lliktemp);
        double diff = abs((oldLLik - mLLik) / mLLik);
        if (diff < mTol)
        {
            GWM_LOG_PROGRESS_PERCENT(exp(- (diff - mTol)));
            break;
        }
        mWt2 = mu;
        itCount++;
    }
    //    return cv;
    return mu;
}
#endif

vec GWRGeneralized::BinomialWtSerial(const mat &x, const vec &y, mat wt)
{
    uword varn = x.n_cols;
    uword dpn = x.n_rows;
    mat betas = mat(varn, dpn, fill::zeros);
    mat S = mat(dpn, dpn);
    mat n = vec(y.n_rows, fill::ones);
    uword itCount = 0;
    //    double lLik = 0.0;
    double oldLLik = 0.0;
    vec mu = vec(dpn, fill::ones) * 0.5;
    vec nu = vec(dpn, fill::zeros);
    //    vec cv = vec(dpn);
    mWt2 = ones(dpn);
    mLLik = 0;
    while (itCount < mMaxiter)
    {
        // 计算公式有调整
        myAdj = nu + (y - mu) / (mu % (1 - mu));
        for (uword i = 0; i < dpn; i++)
        {
            vec wi = wt.col(i);
            vec gwsi = gwPredict(x, myAdj, wi % mWt2);
            betas.col(i) = gwsi;
        }
        mat betas1 = trans(betas);
        nu = Fitted(x, betas1);
        mu = exp(nu) / (1 + exp(nu));
        oldLLik = mLLik;
        mLLik = sum(lchoose(n, y) + (n - y) % log(1 - mu / n) + y % log(mu / n));
        double diff = abs((oldLLik - mLLik) / mLLik);
        if (diff < mTol)
        {
            GWM_LOG_PROGRESS_PERCENT(exp(- (diff - mTol)));
            break;
        }
        mWt2 = n % mu % (1 - mu);
        itCount++;
    }
    return mu;
}

#ifdef ENABLE_OPENMP
vec GWRGeneralized::BinomialWtOmp(const mat &x, const vec &y, mat wt)
{
    uword varn = x.n_cols;
    uword dpn = x.n_rows;
    mat betas = mat(varn, dpn, fill::zeros);
    mat S = mat(dpn, dpn);
    mat n = vec(y.n_rows, fill::ones);
    uword itCount = 0;
    //    double lLik = 0.0;
    double oldLLik = 0.0;
    vec mu = vec(dpn, fill::ones) * 0.5;
    vec nu = vec(dpn, fill::zeros);
    //    vec cv = vec(dpn);
    mWt2 = ones(dpn);
    mLLik = 0;
    while (itCount < mMaxiter)
    {
        // 计算公式有调整
        myAdj = nu + (y - mu) / (mu % (1 - mu));
        for (uword i = 0; i < dpn ; i++)
        {
            vec wi = wt.col(i);
            vec gwsi = gwPredict(x, myAdj, wi % mWt2);
            betas.col(i) = gwsi;
        }
        mat betas1 = trans(betas);
        nu = Fitted(x, betas1);
        mu = exp(nu) / (1 + exp(nu));
        oldLLik = mLLik;
        mLLik = sum(lchoose(n, y) + (n - y) % log(1 - mu / n) + y % log(mu / n));
        double diff = abs((oldLLik - mLLik) / mLLik);
        if (diff < mTol)
        {
            GWM_LOG_PROGRESS_PERCENT(exp(- (diff - mTol)));
            break;
        }
        mWt2 = n % mu % (1 - mu);
        itCount++;
    }
    return mu;
}
#endif

void GWRGeneralized::setBandwidthSelectionCriterionType(const BandwidthSelectionCriterionType &bandwidthSelectionCriterionType)
{
    mBandwidthSelectionCriterionType = bandwidthSelectionCriterionType;
    map<pair<BandwidthSelectionCriterionType, ParallelType>, BandwidthSelectCriterionFunction> mapper = {
        std::make_pair(make_pair(BandwidthSelectionCriterionType::CV, ParallelType::SerialOnly), &GWRGeneralized::bandwidthSizeGGWRCriterionCVSerial),
#ifdef ENABLE_OPENMP
        std::make_pair(make_pair(BandwidthSelectionCriterionType::CV, ParallelType::OpenMP), &GWRGeneralized::bandwidthSizeGGWRCriterionCVOmp),
#endif
        std::make_pair(make_pair(BandwidthSelectionCriterionType::AIC, ParallelType::SerialOnly), &GWRGeneralized::bandwidthSizeGGWRCriterionAICSerial),
#ifdef ENABLE_OPENMP
        std::make_pair(make_pair(BandwidthSelectionCriterionType::AIC, ParallelType::OpenMP), &GWRGeneralized::bandwidthSizeGGWRCriterionAICOmp),
#endif
       };
    mBandwidthSelectCriterionFunction = mapper[make_pair(bandwidthSelectionCriterionType, mParallelType)];
}

void GWRGeneralized::setParallelType(const ParallelType &type)
{
    if (type & parallelAbility())
    {
        mParallelType = type;
        setBandwidthSelectionCriterionType(mBandwidthSelectionCriterionType);
        setFamily(mFamily);
    }
}

mat GWRGeneralized::diag(mat a)
{
    uword n = a.n_rows;
    mat res = mat(uword(0), uword(0));
    if (a.n_cols > 1)
    {
        res = vec(a.n_rows);
        for (uword i = 0; i < n; i++)
        {
            res[i] = a.row(i)[i];
        }
    }
    else
    {
        res = mat(a.n_rows, a.n_rows);
        mat base = eye(n, n);
        for (uword i = 0; i < n; i++)
        {
            res.row(i) = a[i] * base.row(i);
        }
    }
    return res;
}

// GWR clalibration
vec GWRGeneralized::gwPredict(const mat &x, const vec &y, const vec &w)
{
    mat wspan(1, x.n_cols, fill::ones);
    mat xtw = trans(x % (w * wspan));
    mat xtwx = xtw * x;
    mat xtwy = xtw * y;
    mat xtwx_inv = inv(xtwx);
    vec beta = xtwx_inv * xtwy;
    return beta;
}

vec GWRGeneralized::gwFit(const mat &x, const vec &y, const vec &w, uword focus, mat &ci, mat &s_ri)
{
    mat wspan(1, x.n_cols, fill::ones);
    mat xtw = trans(x % (w * wspan));
    mat xtwx = xtw * x;
    mat xtwy = xtw * y;
    mat xtwx_inv = inv(xtwx);
    vec beta = xtwx_inv * xtwy;
    ci = xtwx_inv * xtw;
    s_ri = x.row(focus) * ci;
    return beta;
}

mat GWRGeneralized::dpois(mat y, mat mu)
{
    uword n = y.n_rows;
    mat res = vec(n);
    mat pdf = lgamma(y + 1);
    res = -mu + y % log(mu) - pdf;
    return res;
}

mat GWRGeneralized::lchoose(mat n, mat k)
{
    uword nrow = n.n_rows;
    mat res = vec(nrow);
    //    for(int i = 0;i < nrow; i++){
    //        res.row(i) = lgamma(n[i]+1) - lgamma(n[i]-k[i]+1) - lgamma(k[i]+1);
    //    }
    res = lgamma(n + 1) - lgamma(n - k + 1) - lgamma(k + 1);
    return res;
}

mat GWRGeneralized::dbinom(mat y, mat m, mat mu)
{
    uword n = y.n_rows;
    mat res = vec(n);
    for (uword i = 0; i < n; i++)
    {
        double pdf = gsl_ran_binomial_pdf(int(y[i]), mu[i], int(m[i]));
        res[i] = log(pdf);
    }
    return res;
}

mat GWRGeneralized::lgammafn(mat x)
{
    uword n = x.n_rows;
    mat res = vec(n, fill::zeros);
    for (uword j = 0; j < n; j++)
    {
        res[j] = lgamma(x[j]);
    }
    return res;
}

mat GWRGeneralized::CiMat(const mat &x, const vec &w)
{
    return inv(trans(x) * diagmat(w) * x) * trans(x) * diagmat(w);
}

bool GWRGeneralized::setFamily(Family family)
{
    mFamily = family;
    map<pair<Family, ParallelType>, GGWRfitFunction> mapper = {
        std::make_pair(make_pair(Family::Poisson, ParallelType::SerialOnly), &GWRGeneralized::fitPoissonSerial),
#ifdef ENABLE_OPENMP
        std::make_pair(make_pair(Family::Poisson,
                                 ParallelType::OpenMP),
                       &GWRGeneralized::fitPoissonOmp),
#endif
        std::make_pair(make_pair(Family::Binomial, ParallelType::SerialOnly), &GWRGeneralized::fitBinomialSerial),
#ifdef ENABLE_OPENMP
        std::make_pair(make_pair(Family::Binomial, ParallelType::OpenMP), &GWRGeneralized::fitBinomialOmp),
#endif
    };
    mGGWRfitFunction = mapper[make_pair(family, mParallelType)];
    map<pair<Family, ParallelType>, CalWtFunction> mapper1 = {
        std::make_pair(make_pair(Family::Poisson, ParallelType::SerialOnly), &GWRGeneralized::PoissonWtSerial),
#ifdef ENABLE_OPENMP
        std::make_pair(make_pair(Family::Poisson, ParallelType::OpenMP), &GWRGeneralized::PoissonWtOmp),
#endif
        std::make_pair(make_pair(Family::Binomial, ParallelType::SerialOnly), &GWRGeneralized::BinomialWtSerial),
#ifdef ENABLE_OPENMP
        std::make_pair(make_pair(Family::Binomial, ParallelType::OpenMP), &GWRGeneralized::BinomialWtOmp),
#endif
    };
    mCalWtFunction = mapper1[make_pair(family, mParallelType)];
    return true;
}
