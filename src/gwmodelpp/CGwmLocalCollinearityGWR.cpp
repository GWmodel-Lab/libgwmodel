#include "CGwmLocalCollinearityGWR.h"
#include "CGwmBandwidthSelector.h"
#include "CGwmVariableForwardSelector.h"
#include <assert.h>

#ifdef ENABLE_OPENMP
#include <omp.h>
#endif

using namespace arma;

GwmRegressionDiagnostic CGwmLocalCollinearityGWR::CalcDiagnostic(const mat& x, const vec& y, const mat& betas, const vec& shat)
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

CGwmLocalCollinearityGWR::CGwmLocalCollinearityGWR()
{

}

CGwmLocalCollinearityGWR::~CGwmLocalCollinearityGWR()
{

}

mat CGwmLocalCollinearityGWR::fit()
{
    createDistanceParameter();


    //setXY(mX, mY, mSourceLayer, mDepVar, mIndepVars);
    uword nDp = mCoords.n_rows;
    //选带宽
    //这里判断是否选带宽
    if(mIsAutoselectBandwidth)
    {
        CGwmBandwidthWeight* bw0 = mSpatialWeight.weight<CGwmBandwidthWeight>();
        double lower = bw0->adaptive() ? 20 : 0.0;
        double upper = bw0->adaptive() ? nDp : mSpatialWeight.distance()->maxDistance();
        CGwmBandwidthSelector selector(bw0, lower, upper);
        CGwmBandwidthWeight* bw = selector.optimize(this);
        if (bw)
        {
            mSpatialWeight.setWeight(bw);
            mBandwidthSelectionCriterionList = selector.bandwidthCriterion();
        }
    }
    mat betas(nDp,mX.n_cols,fill::zeros);
    vec localcn(nDp,fill::zeros);
    vec locallambda(nDp,fill::zeros);
    vec hatrow(nDp,fill::zeros);
    //yhat赋值
    mBetas = predict(mX, mY);
    //vec mYHat = fitted(mX,mBetas);
    vec mYHat = sum(mBetas % mX,1);
    vec mResidual = mY - mYHat;
    mDiagnostic.RSS = sum(mResidual % mResidual);
    mDiagnostic.ENP = 2*this->mTrS - this->mTrStS;
    mDiagnostic.EDF = nDp - mDiagnostic.ENP;
    double s2 = mDiagnostic.RSS / (nDp - mDiagnostic.ENP);
    mDiagnostic.AIC = nDp * (log(2*M_PI*s2)+1) + 2*(mDiagnostic.ENP + 1);
    mDiagnostic.AICc = nDp * (log(2*M_PI*s2)) + nDp*( (1+mDiagnostic.ENP/nDp) / (1-(mDiagnostic.ENP+2)/nDp) );
    mDiagnostic.RSquare = 1 - mDiagnostic.RSS/sum((mY - mean(mY)) % (mY - mean(mY)));
    mDiagnostic.RSquareAdjust = 1 - (1 - mDiagnostic.RSquare)*(nDp - 1) / (mDiagnostic.EDF);
    // s2
    // aic 、 aicc
    //调用gwr.lcr.cv.contrib
    //生成诊断信息
    // 诊断
    //mDiagnostic = CalcDiagnostic(mX, mY, mBetas, mSHat);

    return mBetas;
}


void CGwmLocalCollinearityGWR::setBandwidthSelectionCriterion(const BandwidthSelectionCriterionType& criterion)
{
    //setBandwidthSelectionCriterionType
    mBandwidthSelectionCriterion = criterion;
    unordered_map<ParallelType, BandwidthSelectionCriterionCalculator> mapper;
    switch (mBandwidthSelectionCriterion)
    {
    case BandwidthSelectionCriterionType::CV:
        mapper = {
            make_pair(ParallelType::SerialOnly, &CGwmLocalCollinearityGWR::bandwidthSizeCriterionCVSerial),
#ifdef ENABLE_OPENMP
            make_pair(ParallelType::OpenMP, &CGwmLocalCollinearityGWR::bandwidthSizeCriterionCVOmp)
#endif
        };
        break;
    default:
        mapper = {
            make_pair(ParallelType::SerialOnly, &CGwmLocalCollinearityGWR::bandwidthSizeCriterionCVSerial),
#ifdef ENABLE_OPENMP
            make_pair(ParallelType::OpenMP, &CGwmLocalCollinearityGWR::bandwidthSizeCriterionCVOmp)
#endif
        };
        break;
    }
    mBandwidthSelectionCriterionFunction = mapper[mParallelType];
}

vec CGwmLocalCollinearityGWR::ridgelm(const vec& w,double lambda)
{
    //to be done
    //ridgelm
    //X默认加了1
    //默认add.int为False
    mat wspan(1, mX.n_cols, fill::ones);
    mat Xw = mX % (sqrt(w) * wspan);
    mat yw = mY % (sqrt(w));
    //求标准差
    //取mX不含第一列的部分
    mat mXnot1 = mX.cols(1, mX.n_cols - 1);
    //标准差结果矩阵
    mat Xsd(1, mX.n_cols, fill::ones);
    Xsd.cols(1, mX.n_cols - 1) = stddev(mX.cols(1, mX.n_cols - 1),0);
    //Xsd = trans(Xsd);
    //计算Xws
    mat Xws = Xw.each_row() / Xsd;
    double ysd = stddev(yw.col(0));
    mat yws = yw / ysd;
    //计算b值
    //计算crossprod(Xws)
    mat tmpCrossprodXws = trans(Xws)*Xws;
    //生成diag矩阵
    mat tmpdiag = eye(Xws.n_cols,Xws.n_cols);
    //方程求解
    mat tmpXX = tmpCrossprodXws+lambda*tmpdiag;
    mat tmpYY = trans(Xws)*yws;
    vec resultb = inv(tmpXX)*(tmpYY)*ysd/trans(Xsd);
    //如何返回？？
    return resultb;
}

void CGwmLocalCollinearityGWR::setParallelType(const ParallelType& type)
{
    if (type & parallelAbility())
    {
        mParallelType = type;
        switch (type) {
        case ParallelType::SerialOnly:
            mPredictFunction = &CGwmLocalCollinearityGWR::predictSerial;
            break;
#ifdef ENABLE_OPENMP
        case ParallelType::OpenMP:
            mPredictFunction = &CGwmLocalCollinearityGWR::predictOmp;
            break;
#endif
        default:
            mPredictFunction = &CGwmLocalCollinearityGWR::predictSerial;
            break;
        }
    }
    setBandwidthSelectionCriterion(mBandwidthSelectionCriterion);
}

double CGwmLocalCollinearityGWR::bandwidthSizeCriterionCVSerial(CGwmBandwidthWeight* bandwidthWeight)
{
    //行数
    double n = mX.n_rows;
    //列数
    double m = mX.n_cols;
    //初始化矩阵
    mat betas = mat(n,m,fill::zeros);
    vec localcn(n,fill::zeros);
    vec locallambda(n,fill::zeros);
    //取mX不含第一列的部分
    mat mXnot1 = mat(mX.n_rows,mX.n_cols-1,fill::zeros);
    for(uword i=0;i<mX.n_cols-1;i++)
    {
        mXnot1.col(i) = mX.col(i+1);
    }
    //主循环
    for (uword i = 0; i < n ; i++)
    {
        vec distvi = mSpatialWeight.distance()->distance(i);
        vec wgt = bandwidthWeight->weight(distvi);
        //vec wgt = mSpatialWeight.spatialWeight(mDataPoints.row(i),mDataPoints);
        wgt(i) = 0;
        mat wgtspan(1,mXnot1.n_cols,fill::ones);
        mat wgtspan1(1,mX.n_cols,fill::ones);
        //计算xw
        mat xw = mXnot1 % (wgt * wgtspan);
        //计算x1w
        mat x1w = mX % (wgt * wgtspan1);
        //计算用于SVD分解的矩阵
        //计算svd.x
        //mat U,V均为正交矩阵，S为奇异值构成的列向量
        mat U,V;
        colvec S;
        svd(U,S,V,x1w.each_row() / sqrt(sum(x1w % x1w, 0)));
        //赋值操作
        //S.print();
        //qDebug() << S(m);
        localcn(i)=S(0)/S(m-1);
        locallambda(i) = mLambda;
        if(mLambdaAdjust){
            if(localcn(i)>mCnThresh){
                locallambda(i) = (S(0) - mCnThresh*S(m-1)) / (mCnThresh - 1);
            }
        }
        betas.row(i) = trans( ridgelm(wgt,locallambda(i)) );
    }
    //yhat赋值
    //vec mYHat = fitted(mX,betas);
    vec yhat = sum(betas % mX,1);
    //计算residual
    vec residual = mY - yhat;
    //计算cv

    double cv = sum(residual % residual);
    if (isfinite(cv))
    {
        return cv;
    }
    else return DBL_MAX;

}

#ifdef ENABLE_OPENMP
double CGwmLocalCollinearityGWR::bandwidthSizeCriterionCVOmp(CGwmBandwidthWeight* bandwidthWeight)
{
    //行数
    uword n = mX.n_rows;
    //列数
    uword m = mX.n_cols;
    //初始化矩阵
    mat betas = mat(n,m,fill::zeros);
    vec localcn(n,fill::zeros);
    vec locallambda(n,fill::zeros);
    //取mX不含第一列的部分
    mat mXnot1 = mX.cols(1, mX.n_cols - 1);
    //主循环
    uword current = 0;
#pragma omp parallel for num_threads(mOmpThreadNum)
    for (uword i = 0; i < n; i++)
    {
        //int thread = omp_get_thread_num();
        vec distvi = mSpatialWeight.distance()->distance(i);
        vec wgt = bandwidthWeight->weight(distvi);
        //vec wgt = mSpatialWeight.spatialWeight(mDataPoints.row(i),mDataPoints);
        wgt(i) = 0;
        mat wgtspan(1,mXnot1.n_cols,fill::ones);
        mat wgtspan1(1,mX.n_cols,fill::ones);
        //计算xw
        mat xw = mXnot1 % (wgt * wgtspan);
        //计算x1w
        mat x1w = mX % (wgt * wgtspan1);
        //计算用于SVD分解的矩阵
        //计算svd.x
        //mat U,V均为正交矩阵，S为奇异值构成的列向量
        mat U,V;
        colvec S;
        svd(U,S,V,x1w.each_row() / sqrt(sum(x1w % x1w, 0)));
        //赋值操作
        //S.print();
        //qDebug() << S(m);
        localcn(i)=S(0)/S(m-1);
        locallambda(i) = mLambda;
        if(mLambdaAdjust){
            if(localcn(i)>mCnThresh){
                locallambda(i) = (S(0) - mCnThresh*S(m-1)) / (mCnThresh - 1);
            }
        }
        betas.row(i) = trans( ridgelm(wgt,locallambda(i)) );
        current++;

    }
    //yhat赋值
    //vec mYHat = fitted(mX,betas);
    vec yhat = sum(betas % mX,1);
    //计算residual
    vec residual = mY - yhat;
    //计算cv
    double cv = sum(residual % residual);
    if (isfinite(cv))
    {
        return cv;
    }
    else return DBL_MAX;
}
#endif

mat CGwmLocalCollinearityGWR::predictSerial(const mat& x, const vec& y)
{
    uword nRp = mHasPredict ? mPredictData.n_rows : mCoords.n_rows, nVar = mX.n_cols;
    mat betas(nRp, nVar, fill::zeros);
    vec localcn(nRp, fill::zeros);
    vec locallambda(nRp, fill::zeros);
    vec hatrow(nRp, fill::zeros);
    for(uword i=0;i<nRp ;i++)
    {
        vec wi =mSpatialWeight.weightVector(i);
        //计算xw
        //取mX不含第一列的部分
        mat mXnot1 = x.cols(1, x.n_cols - 1);
        mat wispan(1,mXnot1.n_cols,fill::ones);
        mat wispan1(1,x.n_cols,fill::ones);
        //计算xw
        mat xw = mXnot1 % (wi * wispan);
        //计算x1w
        mat x1w = x % (wi * wispan1);
        //计算svd.x
        //mat U,V均为正交矩阵，S为奇异值构成的列向量
        mat U,V;
        colvec S;
        svd(U,S,V,x1w.each_row() / sqrt(sum(x1w % x1w, 0)));
        //qDebug() << mX.n_cols;
        //赋值操作
        localcn(i)=S(0)/S(x.n_cols-1);
        locallambda(i) = mLambda;
        if(mLambdaAdjust){
            if(localcn(i)>mCnThresh){
                locallambda(i) = (S(0) - mCnThresh*S(x.n_cols-1)) / (mCnThresh - 1);
            }
        }
        betas.row(i) = trans(ridgelm(wi,locallambda(i)) );
        //如果没有给regressionpoint
        if(mHasPredict )
        {
            mat xm = x;
            mat xtw = trans(x % (wi * wispan1));
            mat xtwx = xtw * x;
            mat xtwxinv = inv(xtwx);
            rowvec hatrow = x1w.row(i) * xtwxinv * trans(x1w);
            this->mTrS += hatrow(i);
            this->mTrStS += sum(hatrow % hatrow);
        }
    }
    return betas;

}

#ifdef ENABLE_OPENMP
mat CGwmLocalCollinearityGWR::predictOmp(const mat& x, const vec& y)
{
    uword nRp = mHasPredict? mPredictData.n_rows : mCoords.n_rows, nVar = mX.size() + 1;
    mat betas(nRp, nVar, fill::zeros);
    vec localcn(nRp, fill::zeros);
    vec locallambda(nRp, fill::zeros);
    vec hatrow(nRp, fill::zeros);

    mat shat_all(2, mOmpThreadNum, fill::zeros);
    //int current = 0;
#pragma omp parallel for num_threads(mOmpThreadNum)
    for(uword i=0;i < nRp;i++)
    {
        int thread = omp_get_thread_num();
        vec wi = mSpatialWeight.weightVector(i);
        //计算xw
        //取mX不含第一列的部分
        mat mXnot1 = x.cols(1, x.n_cols - 1);
        mat wispan(1,mXnot1.n_cols,fill::ones);
        mat wispan1(1,x.n_cols,fill::ones);
        //计算xw
        mat xw = mXnot1 % (wi * wispan);
        //计算x1w
        mat x1w = x % (wi * wispan1);
        //计算svd.x
        //mat U,V均为正交矩阵，S为奇异值构成的列向量
        mat U,V;
        colvec S;
        svd(U,S,V,x1w.each_row() / sqrt(sum(x1w % x1w, 0)));
        //qDebug() << mX.n_cols;
        //赋值操作
        localcn(i)=S(0)/S(x.n_cols-1);
        locallambda(i) = mLambda;
        if(mLambdaAdjust){
            if(localcn(i)>mCnThresh){
                locallambda(i) = (S(0) - mCnThresh*S(x.n_cols-1)) / (mCnThresh - 1);
            }
        }
        betas.row(i) = trans( ridgelm(wi,locallambda(i)) );
        //如果没有给regressionpoint
        if(mHasPredict)
        {
            mat xm = x;
            mat xtw = trans(x % (wi * wispan1));
            mat xtwx = xtw * x;
            mat xtwxinv = inv(xtwx);
            rowvec hatrow = x1w.row(i) * xtwxinv * trans(x1w);
            shat_all(0, thread) += hatrow(i);
            shat_all(1, thread) += sum(hatrow % hatrow);
            //this->mTrS += hatrow(i);
            //this->mTrStS += sum(hatrow % hatrow);
        }
    }
    vec shat = sum(shat_all,1);
    this->mTrS = sum(shat.row(0));
    this->mTrStS = sum(shat.row(1));
    return betas;
}
#endif


