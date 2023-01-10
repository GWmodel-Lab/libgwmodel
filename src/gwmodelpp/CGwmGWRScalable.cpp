#include "CGwmGWRScalable.h"
#include "CGwmBandwidthSelector.h"
#include "CGwmVariableForwardSelector.h"
#include <assert.h>
#include "GwmLogger.h"

#ifdef ENABLE_OPENMP
#include <omp.h>
#endif

using namespace std;
using namespace arma;
using namespace gwm;

size_t CGwmGWRScalable::treeChildCount = 0;

double CGwmGWRScalable::Loocv(const vec &target, const mat &x, const vec &y, uword poly, const mat &Mx0, const mat &My0)
{
    uword n = x.n_rows, k = x.n_cols, poly1 = poly + 1;
    double b = target(0) * target(0), a = target(1) * target(1);
    vec R0 = vec(poly1, fill::ones) * b;
    for (uword p = 1; p < poly1; p++) {
        R0(p) = pow(b, p + 1);
    }
    R0 = R0 / sum(R0);

    vec Rx(k*k*poly1, fill::zeros), Ry(k*poly1, fill::zeros);
    for (uword p = 0; p < poly1; p++) {
        for (uword k2 = 0; k2 < k; k2++) {
            for (uword k1 = 0; k1 < k; k1++) {
                uword xindex = k1*poly1*k + p*k + k2;
                Rx(xindex) = R0(p);
            }
            uword yindex = p*k + k2;
            Ry(yindex) = R0(p);
        }
    }
    mat Mx = Rx * mat(1, n, fill::ones) % Mx0, My = Ry * mat(1, n, fill::ones) % My0;
    vec yhat(n, 1, fill::zeros);
    for (uword i = 0; i < n; i++) {
        mat sumMx(k, k, fill::zeros);
        vec sumMy(k, fill::zeros);
        for (uword k2 = 0; k2 < k; k2++) {
            for (uword p = 0; p < poly1; p++) {
                for (uword k1 = 0; k1 < k; k1++) {
                    uword xindex = k1*poly1*k + p*k + k2;
                    sumMx(k1, k2) += Mx(xindex, i);
                }
                uword yindex = p*k + k2;
                sumMy(k2) += My(yindex, i);
            }
        }
        sumMx += + a * (x.t() * x);
        sumMy += + a * (x.t() * y);
        try {
            mat beta = solve(sumMx, sumMy);
            yhat.row(i) = x.row(i) * beta;
        } catch (...) {
            return DBL_MAX;
        }
    }
    return sum((y - yhat) % (y - yhat));
}

double CGwmGWRScalable::AICvalue(const vec &target, const mat &x, const vec &y, uword poly, const mat &Mx0, const mat &My0)
{
    uword n = x.n_rows, k = x.n_cols, poly1 = poly + 1;
    double b = target(0) * target(0), a = target(1) * target(1);
    vec R0 = vec(poly1, fill::ones) * b;
    for (uword p = 1; p < poly1; p++) {
        R0(p) = pow(b, p + 1);
    }
    R0 = R0 / sum(R0);

    vec Rx(k*k*poly1, fill::zeros), Ry(k*poly1, fill::zeros);
    for (uword p = 0; p < poly1; p++) {
        for (uword k2 = 0; k2 < k; k2++) {
            for (uword k1 = 0; k1 < k; k1++) {
                uword xindex = k1*poly1*k + p*k + k2;
                Rx(xindex) = R0(p);
            }
            uword yindex = p*k + k2;
            Ry(yindex) = R0(p);
        }
    }
    mat Mx = Rx * mat(1, n, fill::ones) % Mx0, My = Ry * mat(1, n, fill::ones) % My0;
//    mat Mx2 = 2 * a * Mx + ((Rx % Rx) * mat(1, n, fill::ones) % Mx0);

    vec yhat(n, 1, fill::zeros);
    double trS = 0.0/*, trStS = 0.0*/;
    for (uword i = 0; i < n; i++) {
        mat sumMx(k, k, fill::zeros)/*, sumMx2(k, k, fill::zeros)*/;
        vec sumMy(k, fill::zeros);
        for (uword k2 = 0; k2 < k; k2++) {
            for (uword p = 0; p < poly1; p++) {
                for (uword k1 = 0; k1 < k; k1++) {
                    uword xindex = k1*poly1*k + p*k + k2;
                    sumMx(k1, k2) += Mx(xindex, i);
//                    sumMx2(k1, k2) += Mx2(xindex, i);
                }
                uword yindex = p*k + k2;
                sumMy(k2) += My(yindex, i);
            }
        }
        sumMx += a * (x.t() * x);
//        sumMx2 += a * a * (x.t() * x);
        sumMy += a * (x.t() * y);
        if (det(sumMx) < 1e-10) {
            return DBL_MAX;
        } else {
            mat sumMxR = inv(sumMx.t() * sumMx);
            vec trS00 = sumMxR * trans(x.row(i));
            mat trS0 = x.row(i) * trS00;
            trS += trS0[0];

//            vec trStS00 = sumMx2 * trS00;
//            double trStS0 = sum(sum(trS00 * trStS00));
//            trStS += trStS0;

            vec beta = sumMxR * (sumMx.t() * sumMy);
            yhat.row(i) = x.row(i) * beta;
        }
    }
    double sse = sum((y - yhat) % (y - yhat));
    double sig = sqrt(sse / n);
    double AICc = 2 * n * log(sig) + n *log(2*M_PI) +n*(n+trS)/(n-2-trS);
    return isfinite(AICc) ? AICc : DBL_MAX;
}

GwmRegressionDiagnostic CGwmGWRScalable::CalcDiagnostic(const mat& x, const vec& y, const mat& betas, const vec& shat)
{
    vec r = y - sum(betas % x, 1);
    double rss = sum(r % r);
    double n = 1.0 * x.n_rows;
    double AIC = n * log(rss / n) + n * log(2 * datum::pi) + n + shat(0);
    double AICc = n * log(rss / n) + n * log(2 * datum::pi) + n * ((n + shat(0)) / (n - 2 - shat(0)));
    double edf = n - 2 * shat(0) + shat(1);
    double enp = 2 * shat(0) - shat(1);
    double yss = sum((y - mean(y)) % (y - mean(y)));
    double r2 = 1 - rss / yss;
    double r2_adj = 1 - (1 - r2) * (n - 1) / (edf - 1);
    return { rss, AIC, AICc, enp, edf, r2, r2_adj };
}

void CGwmGWRScalable::findDataPointNeighbours()
{
    CGwmBandwidthWeight* bandwidth = mDpSpatialWeight.weight<CGwmBandwidthWeight>();
    uword nDp = mCoords.n_rows, nBw = uword(bandwidth->bandwidth()) < nDp ? uword(bandwidth->bandwidth()) : nDp;
    if (mParameterOptimizeCriterion == BandwidthSelectionCriterionType::CV)
    {
        nBw -= 1;
    }
    umat nnIndex(nBw, nDp, fill::zeros);
    mat nnDists(nBw, nDp, fill::zeros);
    for (uword i = 0; i < nDp; i++)
    {
        vec d = mDpSpatialWeight.distance()->distance(i);
        uvec i_sorted = sort_index(d);
        vec d_sorted = sort(d);
        nnIndex.col(i) = i_sorted(span(0, nBw - 1));
        nnDists.col(i) = d_sorted(span(0, nBw - 1));
    }
    if (mParameterOptimizeCriterion == BandwidthSelectionCriterionType::CV)
    {
        mDpNNDists = trans(nnDists);
        mDpNNIndex = trans(nnIndex);
    }
    else
    {
        mDpNNDists = join_rows(vec(nDp, fill::zeros), trans(nnDists));
        mDpNNIndex = join_rows(linspace<uvec>(0, nDp - 1, nDp), trans(nnIndex));
    }
}

mat CGwmGWRScalable::findNeighbours(const mat& points, umat &nnIndex)
{
    CGwmBandwidthWeight* bandwidth = mSpatialWeight.weight<CGwmBandwidthWeight>();
    uword nDp = mCoords.n_rows;
    uword nRp = points.n_rows;
    uword nBw = uword(bandwidth->bandwidth()) < nDp ? uword(bandwidth->bandwidth()) : nDp;
    umat index(nBw, nRp, fill::zeros);
    mat dists(nBw, nRp, fill::zeros);
    for (uword i = 0; i < nRp; i++)
    {
        vec d = mSpatialWeight.distance()->distance(i);
        uvec i_sorted = sort_index(d);
        vec d_sorted = sort(d);
        index.col(i) = i_sorted(span(0, nBw - 1));
        dists.col(i) = d_sorted(span(0, nBw - 1));
    }
    nnIndex = index.t();
    return dists.t();
}

double scagwr_loocv_multimin_function(const gsl_vector* vars, void* params)
{
    double b_tilde = gsl_vector_get(vars, 0), alpha = gsl_vector_get(vars, 1);
    vec target = { b_tilde, alpha };
    const CGwmGWRScalable::LoocvParams *p = (CGwmGWRScalable::LoocvParams*) params;
    const mat *x = p->x, *y = p->y;
    uword polynomial = p->polynomial;
    const mat *Mx0 = p->Mx0, *My0 = p->My0;
    return CGwmGWRScalable::Loocv(target, *x, *y, polynomial, *Mx0, *My0);
}

double scagwr_aic_multimin_function(const gsl_vector* vars, void* params)
{
    double b_tilde = gsl_vector_get(vars, 0), alpha = gsl_vector_get(vars, 1);
    vec target = { b_tilde, alpha };
    const CGwmGWRScalable::LoocvParams *p = (CGwmGWRScalable::LoocvParams*) params;
    const mat *x = p->x, *y = p->y;
    uword polynomial = p->polynomial;
    const mat *Mx0 = p->Mx0, *My0 = p->My0;
    return CGwmGWRScalable::AICvalue(target, *x, *y, polynomial, *Mx0, *My0);
}

double CGwmGWRScalable::optimize(const mat &Mx0, const mat &My0, double& b_tilde, double& alpha)
{
    gsl_multimin_fminimizer* minizer = gsl_multimin_fminimizer_alloc(gsl_multimin_fminimizer_nmsimplex, 2);
    gsl_vector* target = gsl_vector_alloc(2);
    gsl_vector_set(target, 0, b_tilde);
    gsl_vector_set(target, 1, alpha);
    gsl_vector* step = gsl_vector_alloc(2);
    gsl_vector_set(step, 0, 0.01);
    gsl_vector_set(step, 1, 0.01);
    LoocvParams params = { &mX, &mY, mPolynomial, &Mx0, &My0 };
    gsl_multimin_function function = { mParameterOptimizeCriterion == CV ? &scagwr_loocv_multimin_function : &scagwr_aic_multimin_function, 2, &params };
    double cv = DBL_MAX;
    int status = gsl_multimin_fminimizer_set(minizer, &function, target, step);
    if (status == GSL_SUCCESS)
    {
        size_t iter = 0;
        double size;
        do
        {
            iter++;
            status = gsl_multimin_fminimizer_iterate(minizer);
            if (status) break;
            size = gsl_multimin_fminimizer_size(minizer);
            status = gsl_multimin_test_size(size, 1e-6);
            b_tilde = gsl_vector_get(minizer->x, 0);
            alpha = gsl_vector_get(minizer->x, 1);
            cv = minizer->fval;
        }
        while (status == GSL_CONTINUE && iter < mMaxIter);
        b_tilde = gsl_vector_get(minizer->x, 0);
        alpha = gsl_vector_get(minizer->x, 1);
        cv = minizer->fval;
    }
    gsl_vector_free(target);
    gsl_vector_free(step);
    gsl_multimin_fminimizer_free(minizer);
    return  cv;
}

void CGwmGWRScalable::prepare()
{
//    CGwmBandwidthWeight* bandwidth = mSpatialWeight.weight<CGwmBandwidthWeight>();
    uword knn = mDpNNIndex.n_cols;
    const mat &x = mX, &y = mY, &G0 = mG0;
    uword n = x.n_rows, k = x.n_cols;
    mMx0 = mat((mPolynomial + 1)*k*k, n, fill::zeros);
    mMxx0 = mat((mPolynomial + 1)*k*k, n, fill::zeros);
    mMy0 = mat((mPolynomial + 1)*k, n, fill::zeros);
    mat spanXnei(1, mPolynomial + 1, fill::ones);
    mat spanXtG(1, k, fill::ones);
    for (uword i = 0; i < n; i++) {
        mat G(mPolynomial + 1, knn, fill::ones);
        for (uword p = 0; p < mPolynomial; p++) {
            G.row(p + 1) = pow(G0.row(i), pow(2.0, mPolynomial/2.0)/pow(2.0, p + 1));
        }
        G = trans(G);
        mat xnei = x.rows(mDpNNIndex.row(i));
        vec ynei = y.rows(mDpNNIndex.row(i));
        for (uword k1 = 0; k1 < k; k1++) {
            mat XtG = xnei.col(k1) * spanXnei % G;
            mat XtG2 = xnei.col(k1) * spanXnei % G % G;
            for (uword p = 0; p < (mPolynomial + 1); p++) {
                mat XtGX = XtG.col(p) * spanXtG % xnei;
                mat XtG2X = XtG2.col(p) * spanXtG % xnei;
                for (uword k2 = 0; k2 < k; k2++) {
                    uword xindex = (k1 * (mPolynomial + 1) + p) * k + k2;
                    mMx0(xindex, i) = sum(XtGX.col(k2));
                    mMxx0(xindex, i) = sum(XtG2X.col(k2));
                }
                uword yindex = p * k + k1;
                vec XtGY = XtG.col(p) % ynei;
                mMy0(yindex, i) = sum(XtGY);
            }
        }
    }
}

mat CGwmGWRScalable::predict(const mat& locations)
{
    createDistanceParameter();
    mDpSpatialWeight = mSpatialWeight;
    findDataPointNeighbours();
    CGwmBandwidthWeight* bandwidth = mSpatialWeight.weight<CGwmBandwidthWeight>();
    arma::uword nDp = mX.n_rows, nBw = (uword)bandwidth->bandwidth();
    if (nBw >= nDp) 
    {
        nBw = nDp - 1;
        bandwidth->setBandwidth((double)nBw);
    }
    double band0 = 0.0;
    switch (bandwidth->kernel())
    {
    case CGwmBandwidthWeight::KernelFunctionType::Gaussian:
        band0 = median(mDpNNDists.col(min<uword>(50, nBw) - 1)) / sqrt(3);
        mG0 = exp(-pow(mDpNNDists / band0, 2));
        break;
    case CGwmBandwidthWeight::KernelFunctionType::Exponential:
        band0 = median(mDpNNDists.col(min<uword>(50, nBw) - 1)) / 3;
        mG0 = exp(-pow(mDpNNDists / band0, 2));
        break;
    default:
        return mBetas;
    }
    prepare();
    double b_tilde = 1.0, alpha = 0.01;
    mCV = optimize(mMx0, mMy0, b_tilde, alpha);
    if (mCV < DBL_MAX)
    {
        mScale = b_tilde * b_tilde;
        mPenalty = alpha * alpha;
        mBetas = predictSerial(locations, mX, mY);
    }
    return mBetas;
}

mat CGwmGWRScalable::predictSerial(const mat& locations, const arma::mat &x, const arma::vec &y)
{
    // Create Predict distance parameters
    if (mSpatialWeight.distance()->type() == CGwmDistance::DistanceType::CRSDistance || 
        mSpatialWeight.distance()->type() == CGwmDistance::DistanceType::MinkwoskiDistance)
    {
        mSpatialWeight.distance()->makeParameter({ locations, mCoords });
    }
    CGwmBandwidthWeight* bandwidth = mSpatialWeight.weight<CGwmBandwidthWeight>();
    //uword nDp = mCoords.n_rows;
    uword nRp = locations.n_rows, nVar = mX.n_cols;
    uword nBw = (uword)bandwidth->bandwidth();
    double band0 = 0.0;
    mat G0;
    umat rpNNIndex;
    mat rpNNDists = findNeighbours(locations, rpNNIndex);
    switch (bandwidth->kernel())
    {
    case CGwmBandwidthWeight::KernelFunctionType::Gaussian:
        band0 = median(rpNNDists.col(min<uword>(50, nBw) - 1)) / sqrt(3);
        G0 = exp(-pow(rpNNDists / band0, 2));
        break;
    case CGwmBandwidthWeight::KernelFunctionType::Exponential:
        band0 = median(rpNNDists.col(min<uword>(50, nBw) - 1)) / 3;
        G0 = exp(-pow(rpNNDists / band0, 2));
        break;
    default:
        return mat(nRp, nVar, fill::zeros);
    }

    mMx0 = mat((mPolynomial + 1)*nVar*nVar, nRp, fill::zeros);
    mMxx0 = mat((mPolynomial + 1)*nVar*nVar, nRp, fill::zeros);
    mMy0 = mat((mPolynomial + 1)*nVar, nRp, fill::zeros);
    mat spanXnei(1, mPolynomial + 1, fill::ones);
    mat spanXtG(1, nVar, fill::ones);
    for (arma::uword i = 0; i < nRp; i++)
    {
        mat G(mPolynomial + 1, nBw, fill::ones);
        for (uword p = 0; p < mPolynomial; p++) {
            G.row(p + 1) = pow(G0.row(i), pow(2.0, mPolynomial/2.0)/pow(2.0, p + 1));
        }
        G = trans(G);
        mat xnei = x.rows(rpNNIndex.row(i));
        vec ynei = y.rows(rpNNIndex.row(i));
        for (arma::uword k1 = 0; k1 < nVar; k1++) {
            mat XtG = xnei.col(k1) * spanXnei % G;
            mat XtG2 = xnei.col(k1) * spanXnei % G % G;
            for (uword p = 0; p < (mPolynomial + 1); p++) {
                mat XtGX = XtG.col(p) * spanXtG % xnei;
                mat XtG2X = XtG2.col(p) * spanXtG % xnei;
                for (arma::uword k2 = 0; k2 < nVar; k2++) {
                    uword xindex = (k1 * (mPolynomial + 1) + p) * nVar + k2;
                    mMx0(xindex, i) = sum(XtGX.col(k2));
                    mMxx0(xindex, i) = sum(XtG2X.col(k2));
                }
                uword yindex = p * nVar + k1;
                vec XtGY = XtG.col(p) % ynei;
                mMy0(yindex, i) = sum(XtGY);
            }
        }
    }

    uword poly1 = mPolynomial + 1;
    double b = mScale, a = mPenalty;
    vec R0 = vec(poly1, fill::ones) * b;
    for (uword p = 1; p < poly1; p++) {
        R0(p) = pow(b, p + 1);
    }
    R0 = R0 / sum(R0);
    vec Rx(nVar*nVar*poly1, fill::zeros), Ry(nVar*poly1, fill::zeros);
    for (uword p = 0; p < poly1; p++) {
        for (uword k2 = 0; k2 < nVar; k2++) {
            for (uword k1 = 0; k1 < nVar; k1++) {
                uword xindex = k1*poly1*nVar + p*nVar + k2;
                Rx(xindex) = R0(p);
            }
            uword yindex = p*nVar + k2;
            Ry(yindex) = R0(p);
        }
    }
    mat Mx = Rx * mat(1, nRp, fill::ones) % mMx0, My = Ry * mat(1, nRp, fill::ones) % mMy0;
    mat Mx2 = 2 * a * Mx + ((Rx % Rx) * mat(1, nRp, fill::ones) % mMx0);

    mat betas(nVar, nRp, fill::zeros);
    for (uword i = 0; i < nRp; i++) {
        mat sumMx(nVar, nVar, fill::zeros), sumMx2(nVar, nVar, fill::zeros);
        vec sumMy(nVar, fill::zeros);
        for (uword k2 = 0; k2 < nVar; k2++) {
            for (uword p = 0; p < poly1; p++) {
                for (uword k1 = 0; k1 < nVar; k1++) {
                    uword xindex = k1*poly1*nVar + p*nVar + k2;
                    sumMx(k1, k2) += Mx(xindex, i);
                    sumMx2(k1, k2) += Mx2(xindex, i);
                }
                uword yindex = p*nVar + k2;
                sumMy(k2) += My(yindex, i);
            }
        }
        sumMx += a * (x.t() * x);
        sumMx2 += a * a * (x.t() * x);
        sumMy += a * (x.t() * y);
        mat sumMxR = inv(sumMx.t() * sumMx);
        betas.col(i) = sumMxR * (sumMx.t() * sumMy);
    }

    return betas.t();
}

mat CGwmGWRScalable::fit()
{
    createDistanceParameter();
    mDpSpatialWeight = mSpatialWeight;
    findDataPointNeighbours();
    CGwmBandwidthWeight* bandwidth = mSpatialWeight.weight<CGwmBandwidthWeight>();
    arma::uword nDp = mX.n_rows, nBw = (uword)bandwidth->bandwidth();
    if (nBw >= nDp) 
    {
        nBw = nDp - 1;
        bandwidth->setBandwidth((double)nBw);
    }
    double band0 = 0.0;
    switch (bandwidth->kernel())
    {
    case CGwmBandwidthWeight::KernelFunctionType::Gaussian:
        band0 = median(mDpNNDists.col(min<uword>(50, nBw) - 1)) / sqrt(3);
        mG0 = exp(-pow(mDpNNDists / band0, 2));
        break;
    case CGwmBandwidthWeight::KernelFunctionType::Exponential:
        band0 = median(mDpNNDists.col(min<uword>(50, nBw) - 1)) / 3;
        mG0 = exp(-pow(mDpNNDists / band0, 2));
        break;
    default:
        return mBetas;
    }
    prepare();
    double b_tilde = 1.0, alpha = 0.01;
    mCV = optimize(mMx0, mMy0, b_tilde, alpha);
    if (mCV < DBL_MAX)
    {
        mScale = b_tilde * b_tilde;
        mPenalty = alpha * alpha;
        mBetas = fitSerial(mX, mY);
        mDiagnostic = CalcDiagnostic( mX,mY, mBetas, mShat);
        double trS = mShat(0), trStS = mShat(1);
        double sigmaHat = mDiagnostic.RSS / (nDp - 2 * trS + trStS);
        vec yhat = sum(mX % mBetas, 1);
        vec residual = mY - yhat;
        mBetasSE = sqrt(sigmaHat * mBetasSE);
        mat betasTV = mBetas / mBetasSE;
        
    }
    return mBetas;
}

arma::mat CGwmGWRScalable::fitSerial(const arma::mat &x, const arma::vec &y)
{
    CGwmBandwidthWeight* bandwidth = mSpatialWeight.weight<CGwmBandwidthWeight>();
    uword bw = (uword)bandwidth->bandwidth();
    uword n = x.n_rows, k = x.n_cols, poly1 = mPolynomial + 1;
    if (bw >= n)
    {
        bw = bw - 1;
        bandwidth->setBandwidth((double)bw);
    }
    double b = mScale, a = mPenalty;
    mat XtX = x.t() * x, XtY = x.t() * y;
    mat betas(k, n, fill::zeros);

    double band0 = 0.0;
    umat dpNNIndex;
    mat dpNNDists = findNeighbours(mCoords, dpNNIndex);
    switch (bandwidth->kernel())
    {
    case CGwmBandwidthWeight::KernelFunctionType::Gaussian:
        band0 = median(dpNNDists.col(min<uword>(50, bw) - 1)) / sqrt(3);
        mG0 = exp(-pow(dpNNDists / band0, 2));
        break;
    case CGwmBandwidthWeight::KernelFunctionType::Exponential:
        band0 = median(dpNNDists.col(min<uword>(50, bw) - 1)) / 3;
        mG0 = exp(-pow(dpNNDists / band0, 2));
        break;
    default:
        return betas.t();
    }

    mMx0 = mat((mPolynomial + 1)*k*k, n, fill::zeros);
    mMxx0 = mat((mPolynomial + 1)*k*k, n, fill::zeros);
    mMy0 = mat((mPolynomial + 1)*k, n, fill::zeros);
    mat spanXnei(1, mPolynomial + 1, fill::ones);
    mat spanXtG(1, k, fill::ones);
    for (uword i = 0; i < n; i++) {
        mat G(mPolynomial + 1, bw, fill::ones);
        for (uword p = 0; p < mPolynomial; p++) {
            G.row(p + 1) = pow(mG0.row(i), pow(2.0, mPolynomial/2.0)/pow(2.0, p + 1));
        }
        G = trans(G);
        mat xnei = x.rows(dpNNIndex.row(i));
        vec ynei = y.rows(dpNNIndex.row(i));
        for (uword k1 = 0; k1 < k; k1++) {
            mat XtG = xnei.col(k1) * spanXnei % G;
            mat XtG2 = xnei.col(k1) * spanXnei % G % G;
            for (uword p = 0; p < (mPolynomial + 1); p++) {
                mat XtGX = XtG.col(p) * spanXtG % xnei;
                mat XtG2X = XtG2.col(p) * spanXtG % xnei;
                for (uword k2 = 0; k2 < k; k2++) {
                    uword xindex = (k1 * (mPolynomial + 1) + p) * k + k2;
                    mMx0(xindex, i) = sum(XtGX.col(k2));
                    mMxx0(xindex, i) = sum(XtG2X.col(k2));
                }
                uword yindex = p * k + k1;
                vec XtGY = XtG.col(p) % ynei;
                mMy0(yindex, i) = sum(XtGY);
            }
        }
    }

    vec R0 = vec(poly1, fill::ones) * b;
    for (uword p = 1; p < poly1; p++) {
        R0(p) = pow(b, p + 1);
    }
    R0 = R0 / sum(R0);
    vec Rx(k*k*poly1, fill::zeros), Ry(k*poly1, fill::zeros);
    for (uword p = 0; p < poly1; p++) {
        for (uword k2 = 0; k2 < k; k2++) {
            for (uword k1 = 0; k1 < k; k1++) {
                uword xindex = k1*poly1*k + p*k + k2;
                Rx(xindex) = R0(p);
            }
            uword yindex = p*k + k2;
            Ry(yindex) = R0(p);
        }
    }
    mat Mx = Rx * mat(1, n, fill::ones) % mMx0, My = Ry * mat(1, n, fill::ones) % mMy0;
    mat Mx2 = 2 * a * Mx + ((Rx % Rx) * mat(1, n, fill::ones) % mMxx0);

    mat bse(k, n, fill::zeros);
    double trS = 0.0, trStS = 0.0;
    for (uword i = 0; i < n; i++) {
        mat sumMx(k, k, fill::zeros), sumMx2(k, k, fill::zeros);
        vec sumMy(k, fill::zeros);
        for (uword k2 = 0; k2 < k; k2++) {
            for (uword p = 0; p < poly1; p++) {
                for (uword k1 = 0; k1 < k; k1++) {
                    uword xindex = k1*poly1*k + p*k + k2;
                    sumMx(k1, k2) += Mx(xindex, i);
                    sumMx2(k1, k2) += Mx2(xindex, i);
                }
                uword yindex = p*k + k2;
                sumMy(k2) += My(yindex, i);
            }
        }
        sumMx += a * (x.t() * x);
        sumMx2 += a * a * (x.t() * x);
        sumMy += a * (x.t() * y);
        try {
            mat sumMxR = inv(trans(sumMx) * sumMx);
            vec trS00 = sumMxR * trans(x.row(i));
            mat trS0 = x.row(i) * trS00;
            trS += trS0[0];

            vec trStS00 = sumMx2 * trS00;
            double trStS0 = sum(trS00 % trStS00);
            trStS += trStS0;

            vec beta = sumMxR * (sumMx.t() * sumMy);
            betas.col(i) = beta;

            mat bse00 = sumMxR * sumMx2 * sumMxR;
            vec bse0 = sqrt(diagvec(bse00));
            bse.col(i) = bse0;
        } catch (const exception& e) {
            GWM_LOG_ERROR(e.what());
        }
    }
    mBetasSE = bse.t();
    mShat = { trS, trStS };
    return betas.t();
}

bool CGwmGWRScalable::isValid()
{
    if (CGwmGWRBase::isValid())
    {
        CGwmBandwidthWeight* bw = mSpatialWeight.weight<CGwmBandwidthWeight>();
        if (!(bw->kernel() == CGwmBandwidthWeight::Gaussian || bw->kernel() == CGwmBandwidthWeight::Exponential))
            return false;

        if (!(bw->bandwidth()>0))
        {
            return false;
        }

        return true;
    }
    else return false;
}
