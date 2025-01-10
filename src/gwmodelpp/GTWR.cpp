#include "GTWR.h"
#include "BandwidthSelector.h"
#include "VariableForwardSelector.h"
#include "Logger.h"

#ifdef ENABLE_OPENMP
#include <omp.h>
#endif

using namespace std;
using namespace arma;
using namespace gwm;

RegressionDiagnostic GTWR::CalcDiagnostic(const mat& x, const vec& y, const mat& betas, const vec& shat)
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

mat GTWR::fit()
{
    GWM_LOG_STAGE("Initializing")
    createDistanceParameter();
    uword nDp = mCoords.n_rows, nVars = mX.n_cols;
    GWM_LOG_STOP_RETURN(mStatus, mat(nDp, nVars, arma::fill::zeros));

    if(mIsAutoselectLambdaBw){
        GWM_LOG_STAGE("Lambda and bandwidth optimization")
        BandwidthWeight *bw0 = mSpatialWeight.weight<BandwidthWeight>();
        mStdistance = mSpatialWeight.distance<CRSSTDistance>();
        // double lambda0 = 0.05;
        // double first_bw= bw0->adaptive() ? nDp*0.618 : 0.0;
        // bw0->setBandwidth(first_bw);
        vec optim = vec(2, fill::zeros);
        optim = lambdaBwAutoSelection(bw0, 1000, 1e-3);

        mStdistance->setLambda(optim(0));
        bw0->setBandwidth(optim(1));
        //不能再进行后面的优化
        mIsAutoselectBandwidth=false;
        mIsAutoselectLambda=false;
    }

    if (mIsAutoselectBandwidth)
    {
        GWM_LOG_STAGE("Bandwidth optimization")
        BandwidthWeight *bw0 = mSpatialWeight.weight<BandwidthWeight>();
        double lower = bw0->adaptive() ? 20 : 0.0;
        double upper = bw0->adaptive() ? nDp : mSpatialWeight.distance()->maxDistance();

        GWM_LOG_INFO(IBandwidthSelectable::infoBandwidthCriterion(bw0));
        BandwidthSelector selector(bw0, lower, upper);
        BandwidthWeight *bw = selector.optimize(this);
        if (bw)
        {
            mSpatialWeight.setWeight(bw);
            mBandwidthSelectionCriterionList = selector.bandwidthCriterion();
        }
        GWM_LOG_STOP_RETURN(mStatus, mat(nDp, nVars, arma::fill::zeros));
    }

    if (mIsAutoselectLambda)
    {
        GWM_LOG_STAGE("Lambda optimization")
        BandwidthWeight *bw = mSpatialWeight.weight<BandwidthWeight>();
        mStdistance = mSpatialWeight.distance<CRSSTDistance>();

        GWM_LOG_INFO(infoLambdaCriterion());
        double lambda = lambdaAutoSelection(bw);
        mStdistance->setLambda(lambda);
        GWM_LOG_STOP_RETURN(mStatus, mat(nDp, nVars, arma::fill::zeros));
    }

    GWM_LOG_STAGE("Model fitting")
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

mat GTWR::predict(const mat& locations)
{
    createPredictionDistanceParameter(locations);
    mBetas = (this->*mPredictFunction)(locations, mX, mY);
    return mBetas;
}

void GTWR::createPredictionDistanceParameter(const arma::mat& locations)
{
    if (mSpatialWeight.distance()->type() == Distance::DistanceType::CRSSTDistance)
    {
        mSpatialWeight.distance()->makeParameter({ mCoords, mCoords, vTimes, vTimes });
    }
    mStdistance = mSpatialWeight.distance<CRSSTDistance>();
}

void GTWR::createDistanceParameter()
{
    if (mSpatialWeight.distance()->type() == Distance::DistanceType::CRSSTDistance)
    {
        mSpatialWeight.distance()->makeParameter({ mCoords, mCoords, vTimes, vTimes });
    }
    mStdistance = mSpatialWeight.distance<CRSSTDistance>();
}


mat GTWR::predictSerial(const mat& locations, const mat& x, const vec& y)
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

mat GTWR::fitSerial(const mat& x, const vec& y, mat& betasSE, vec& shat, vec& qDiag, mat& S)
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

double GTWR::bandwidthSizeCriterionCVSerial(BandwidthWeight* bandwidthWeight)
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
        GWM_LOG_PROGRESS_PERCENT(exp(- abs(mBandwidthLastCriterion - cv)));
        mBandwidthLastCriterion = cv;
        return cv;
    }
    else return DBL_MAX;
}

double GTWR::bandwidthSizeCriterionAICSerial(BandwidthWeight* bandwidthWeight)
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
        GWM_LOG_PROGRESS_PERCENT(exp(- abs(mBandwidthLastCriterion - value)));
        mBandwidthLastCriterion = value;
        return value;
    }
    else return DBL_MAX;
}

void GTWR::setBandwidthSelectionCriterion(const BandwidthSelectionCriterionType& criterion)
{
    mBandwidthSelectionCriterion = criterion;
    unordered_map<BandwidthSelectionCriterionType, BandwidthSelectionCriterionCalculator> mapper;
    mapper = {
        make_pair(BandwidthSelectionCriterionType::CV, &GTWR::bandwidthSizeCriterionCVSerial),
        make_pair(BandwidthSelectionCriterionType::AIC, &GTWR::bandwidthSizeCriterionAICSerial)
    };
    mBandwidthSelectionCriterionFunction = mapper[mBandwidthSelectionCriterion];
}

bool GTWR::isValid()
{
    if (GWRBase::isValid())
    {
        double bw = mSpatialWeight.weight<BandwidthWeight>()->bandwidth();
        if (!(bw > 0))
        {
            return false;
        }
        double lambda=mStdistance->lambda();
        if (lambda < 0 || lambda > 1)
        {
            return false;
        }
        return true;
    }
    else return false;
}

#ifdef ENABLE_OPENMP
mat GTWR::predictOmp(const mat& locations, const mat& x, const vec& y)
{
    uword nRp = locations.n_rows, nVar = x.n_cols;
    mat betas(nVar, nRp, arma::fill::zeros);
    bool success = true;
    std::exception except;
#pragma omp parallel for num_threads(mOmpThreadNum)
    for (uword i = 0; i < nRp; i++)
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

mat GTWR::fitOmp(const mat& x, const vec& y, mat& betasSE, vec& shat, vec& qDiag, mat& S)
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
    for (uword i = 0; i < nDp; i++)
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


double GTWR::bandwidthSizeCriterionCVOmp(BandwidthWeight* bandwidthWeight)
{
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
        GWM_LOG_PROGRESS_PERCENT(exp(- abs(mBandwidthLastCriterion - cv)));
        mBandwidthLastCriterion = cv;
        return cv;
    }
    else return DBL_MAX;
}

double GTWR::bandwidthSizeCriterionAICOmp(BandwidthWeight* bandwidthWeight)
{
    uword nDp = mCoords.n_rows, nVar = mX.n_cols;
    mat betas(nVar, nDp, fill::zeros);
    mat shat_all(2, mOmpThreadNum, fill::zeros);
    bool flag = true;
#pragma omp parallel for num_threads(mOmpThreadNum)
    for (uword i = 0; i < nDp; i++)
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
            GWM_LOG_PROGRESS_PERCENT(exp(- abs(mBandwidthLastCriterion - value)));
            mBandwidthLastCriterion = value;
            return value;
        }
        else return DBL_MAX;
    }
    else return DBL_MAX;
}
#endif

void GTWR::setParallelType(const ParallelType& type)
{
    if (type & parallelAbility())
    {
        mParallelType = type;
        switch (type) {
        case ParallelType::SerialOnly:
            mPredictFunction = &GTWR::predictSerial;
            mFitFunction = &GTWR::fitSerial;
            break;
#ifdef ENABLE_OPENMP
        case ParallelType::OpenMP:
            mPredictFunction = &GTWR::predictOmp;
            mFitFunction = &GTWR::fitOmp;
            break;
#endif
        default:
            mPredictFunction = &GTWR::predictSerial;
            mFitFunction = &GTWR::fitSerial;
            break;
        }
        setBandwidthSelectionCriterion(mBandwidthSelectionCriterion);
    }
}

Status GTWR::r_squareByLambda(BandwidthWeight* bandwidthWeight, double lambda, double& rsquare)
{
    // (void)(params);
    double r2;
    uword nDp = mCoords.n_rows, nVar = mX.n_cols;
    mat betas(nVar, nDp, fill::zeros);
    mStdistance->setLambda(lambda);
    mStdistance->makeParameter({ mCoords, mCoords, vTimes, vTimes });
    for (uword i = 0; i < nDp; i++)
    {
        GWM_LOG_STOP_BREAK(mStatus);
        mat d = mStdistance->distance(i);
        // vec w = mSpatialWeight.weightVector(i);
        vec w = bandwidthWeight->weight(d);
        mat xtw = trans(mX.each_col() % w);
        mat xtwx = xtw * mX;
        mat xtwy = xtw * mY;
        try
        {
            mat xtwx_inv = inv_sympd(xtwx);
            betas.col(i) = xtwx_inv * xtwy;
        }
        catch (const exception &e)
        {
            GWM_LOG_ERROR(e.what());
            throw e;
        }
    }
    if (mStatus == Status::Success){
        betas = betas.t();
        vec r = mY - sum(betas % mX, 1);
        double rss = sum(r % r);
        double yss = sum((mY - mean(mY)) % (mY - mean(mY)));
        r2 = 1 - rss / yss;
        rsquare = r2;
        GWM_LOG_INFO(infoLambdaCriterion(lambda, rsquare));
    }
    else
    {
        rsquare = 0.0;
    }
    return mStatus;
}

double GTWR::lambdaAutoSelection(BandwidthWeight* bw)
{
    // int status;
    // int iter = 0, max_iter = 1000;
    // const gsl_min_fminimizer_type *T;
    // gsl_min_fminimizer *s;
    // double mini = 0.05, lower = 0.0, upper = 1.0;
    // gsl_function F;
    // F.function = &GTWR::r_squareByLambda;//this make error
    // T = gsl_min_fminimizer_brent;
    // s = gsl_min_fminimizer_alloc(T);
    // gsl_min_fminimizer_set(s, &F, mini, lower, upper);
    // do {
    //     iter++;
    //     status = gsl_min_fminimizer_iterate(s);
    //     mini = gsl_min_fminimizer_x_minimum(s);
    //     lower = gsl_min_fminimizer_x_lower(s);
    //     upper = gsl_min_fminimizer_x_upper(s);
    //     status = gsl_min_test_interval(lower, upper, 0.001, 0.0);
    //     if (status == GSL_SUCCESS)
    //         printf("Converged:\n");

    //     printf("%5d [%.7f, %.7f] %.7f %+.7f\n",iter, lower, upper,mini, upper - lower);
    // } while (status == GSL_CONTINUE && iter < max_iter);
    // gsl_min_fminimizer_free(s);
    // return mini;
    int iter = 0, max_iter = 100;
    double eps = 1e-12;
    double ratio = (sqrt(5) - 1) / 2;
    // [a, p, q, b]
    double a = 0.0, b = 1.0;
    double step = b - a;
    double p = a + (1 - ratio) * step, q = a + ratio * step;
    double f_a, f_b, f_p, f_q;
    Status s_a = r_squareByLambda(bw, a, f_a);
    Status s_b = r_squareByLambda(bw, b, f_b);
    Status s_p = r_squareByLambda(bw, p, f_p);
    Status s_q = r_squareByLambda(bw, q, f_q);
    umat sm = { (u64)s_a, (u64)s_b, (u64)s_p, (u64)s_q };
    // r方越大越好
    while (all(vectorise(sm) == 0) && abs(f_a - f_b) >= eps && iter < max_iter)
    {
        if (f_p > f_q)
        {
            b = q; f_b = f_q;
            q = p; f_q = f_p;
            step = b - a;
            p = a + (1 - ratio) * step;
            sm(2) = u64(r_squareByLambda(bw, p, f_p));
        }
        else
        {
            a = p; f_a = f_p;
            p = q; f_p = f_q;
            step = b - a;
            q = a + ratio * step;
            sm(3) = u64(r_squareByLambda(bw, q, f_q));
        }
        iter++;
    }
    if (all(vectorise(sm) == u64(Status::Success)))
    {
        double golden = (b + a) / 2;
        return golden;
    }
    else return 0.0;
}

double GTWR::criterionByLambdaBw(BandwidthWeight *bandwidth, double lambda, BandwidthSelectionCriterionType criterion)
{
    uword nDp = mCoords.n_rows, nVar = mX.n_cols;
    mat betas(nVar, nDp, fill::zeros);
    vec shat(2, fill::zeros);
    double value = 0.0;
    mStdistance->setLambda(lambda);
    mStdistance->makeParameter({mCoords, mCoords, vTimes, vTimes});
    //要优化的可变带宽此前被归一化了
    if(bandwidth->adaptive()){
        bandwidth->setBandwidth(bandwidth->bandwidth() * nDp);
    }else{
        bandwidth->setBandwidth(bandwidth->bandwidth() * mStdistance->maxDistance());
    }
    for (uword i = 0; i < nDp; i++)
    {
        GWM_LOG_STOP_BREAK(mStatus);
        vec d = mSpatialWeight.distance()->distance(i);
        vec w = bandwidth->weight(d);
        if (criterion == BandwidthSelectionCriterionType::CV)
        {
            w(i) = 0.0;
        }
        mat xtw = trans(mX.each_col() % w);
        mat xtwx = xtw * mX;
        mat xtwy = xtw * mY;
        switch (criterion)
        {
        case BandwidthSelectionCriterionType::AIC:
            try
            {
                mat xtwx_inv = inv_sympd(xtwx);
                betas.col(i) = xtwx_inv * xtwy;
                mat ci = xtwx_inv * xtw;
                mat si = mX.row(i) * ci;
                shat(0) += si(0, i);
                shat(1) += det(si * si.t());
            }
            catch (const exception &e)
            {
                GWM_LOG_ERROR(e.what());
                return DBL_MAX;
            }
            break;
        case BandwidthSelectionCriterionType::CV:
            try
            {
                mat xtwx_inv = inv_sympd(xtwx);
                vec beta = xtwx_inv * xtwy;
                double res = mY(i) - det(mX.row(i) * beta);
                value += res * res;
            }
            catch (const exception &e)
            {
                GWM_LOG_ERROR(e.what());
                return DBL_MAX;
            }
            break;
        default:
            try
            {
                mat xtwx_inv = inv_sympd(xtwx);
                betas.col(i) = xtwx_inv * xtwy;
                mat ci = xtwx_inv * xtw;
                mat si = mX.row(i) * ci;
                shat(0) += si(0, i);
                shat(1) += det(si * si.t());
            }
            catch (const exception &e)
            {
                GWM_LOG_ERROR(e.what());
                return DBL_MAX;
            }
            break;
        }
    }
    if (criterion == BandwidthSelectionCriterionType::AIC)
    {
        value = GWRBase::AICc(mX, mY, betas.t(), shat);
    }
    if (mStatus == Status::Success && isfinite(value))
    {
        GWM_LOG_PROGRESS_PERCENT(exp(-abs(mBandwidthLastCriterion - value)));
        mBandwidthLastCriterion = value;
        return value;
    }
    else
        return DBL_MAX;
}

double GTWR::criterion_function (const gsl_vector *target, void *params)
{
    // 将 void* 转换为 Parameter* 类型
    GTWR::Parameter* p = static_cast<Parameter*>(params);

    // 获取参数指针
    GTWR* instance = p->instance;
    BandwidthWeight* bandwidth = p->bandwidth;
    // double lambda = p->lambda;
    BandwidthSelectionCriterionType criterionType=instance->mBandwidthSelectionCriterion;

    // 从 gsl_vector 更新
    double lambda_value = gsl_vector_get(target, 0);
    double bandwidth_value = gsl_vector_get(target, 1);

    //问题：如果优化的时候reset lambda，设置的值不在0-1范围里，怎么办
    // instance->mStdistance->setLambda(lambda_value);//这里和后面计算的里面重复了
    bandwidth->setBandwidth(bandwidth_value);
    return instance->criterionByLambdaBw(bandwidth, lambda_value, criterionType);
}

vec GTWR::lambdaBwAutoSelection(BandwidthWeight* bandwidth, size_t max_iter, double min_step)
{
    vec optim(2, fill::zeros);
    uword nDp = mCoords.n_rows;
    gsl_multimin_fminimizer *minimizer = gsl_multimin_fminimizer_alloc(gsl_multimin_fminimizer_nmsimplex2rand, 2);
    gsl_vector* lambda_bw = gsl_vector_alloc(2);
    gsl_vector* steps = gsl_vector_alloc(2);
    gsl_vector_set(lambda_bw, 0, this->mStdistance->lambda());
    //带宽近似映射到了0-1
    // gsl_vector_set(lambda_bw, 1, bandwidth->adaptive() ? 0.618 : bandwidth->bandwidth());
    double mdis=mStdistance->maxDistance();
    gsl_vector_set(lambda_bw, 1, bandwidth->adaptive() ? bandwidth->bandwidth() / double(nDp) : bandwidth->bandwidth() / mdis);
    // gsl_vector_set(lambda_bw, 1, bandwidth->bandwidth());
    gsl_vector_set_all(steps, min_step);

    GTWR::Parameter params;
    params.instance = this;  // GTWR类的实例是当前对象
    params.bandwidth = bandwidth;
    params.lambda = this->mStdistance->lambda();
    

    // gsl_multimin_function minex_func;
    gsl_multimin_function minex_func = { criterion_function, 2, &params };

    int status = gsl_multimin_fminimizer_set(minimizer, &minex_func, lambda_bw, steps);

    if (status == GSL_SUCCESS)
    {
        size_t iter = 0;
        double size = DBL_MAX;
        do
        {
            iter++;
            status = gsl_multimin_fminimizer_iterate(minimizer);
            if (status && params.instance->status() != Status::Success)
                break;
            size = gsl_multimin_fminimizer_size(minimizer);
            status = gsl_multimin_test_size(size, min_step/1000);
            // cout<<"lambda:"<< abs(gsl_vector_get(minimizer->x, 0))<<endl;
            // cout<<"bandwidth:"<< abs(gsl_vector_get(minimizer->x, 1))<<endl;

            // #ifdef _DEBUG
            // stringstream sDebug;
            // for (size_t m = 0; m < 2; m++)
            // {
            //     sDebug << gsl_vector_get(minimizer->x, m) << ",";
            // }
            // sDebug << minimizer->fval << ",";
            // sDebug << size;
            // params.instance->debug(sDebug.str(), __FUNCTION__, __FILE__);
            // #endif
        } 
        while (status == GSL_CONTINUE && iter < max_iter);
        // #ifdef _DEBUG
        // stringstream sDebug;
        // for (size_t m = 0; m < 2; m++)
        // {
        //     sDebug << gsl_vector_get(minimizer->x, m) << ",";
        // }
        // sDebug << minimizer->fval << ",";
        // sDebug << size;
        // params.instance->debug(sDebug.str(), __FUNCTION__, __FILE__);
        // #endif
        double optbw=bandwidth->adaptive() ? abs(gsl_vector_get(minimizer->x, 1)) * nDp : abs(gsl_vector_get(minimizer->x, 1)) * mdis;
        cout<<"optimezed lambda:"<< abs(gsl_vector_get(minimizer->x, 0))<<endl;
        cout<<"optimezed bandwidth:"<< optbw <<endl;
        optim(0)=abs(gsl_vector_get(minimizer->x, 0));
        optim(1)=optbw;
    }

    gsl_vector_free(lambda_bw);
    gsl_vector_free(steps);
    gsl_multimin_fminimizer_free(minimizer);

    return optim;
}