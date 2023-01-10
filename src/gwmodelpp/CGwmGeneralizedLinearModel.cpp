#include "CGwmGeneralizedLinearModel.h"
#include "CGwmPoissonModel.h"
#include "CGwmBinomialModel.h"
//#include "GWmodel.h"
#include "CGwmGGWR.h"

using namespace arma;

CGwmGeneralizedLinearModel::CGwmGeneralizedLinearModel()
{
    mWeight = vec(uword(0));
    mOffset = vec(uword(0));
    mIntercept = true;
    mMaxit = 25;
    mEpsilon = 1e-8;
}

void CGwmGeneralizedLinearModel::fit(){
    uword nVars = mX.n_cols, nObs = mY.n_rows;
    bool empty = (nVars == 0);
    mat coefold;
    mat eta = vec(uword(0));
    mat mu;
    if(mWeight.is_empty() ){
        mWeight = ones(nObs);
    }
    if(mOffset.is_empty() ){
        mOffset = zeros(nObs);
    }
    if(mFamily == CGwmGGWR::Family::Poisson){
        //初始化模型
        mModel = new CGwmPoissonModel();
        mModel->setY(mY);
        mModel->setWeight(mWeight);
    }
    else{
        mModel = new CGwmBinomialModel();
        mModel->setY(mY);
        mModel->setWeight(mWeight);
    }
    mat n = mModel->initialize();
    mMuStart = mModel->muStart();
    if(empty ){
        eta = zeros(nObs) + mOffset;
        mu = mModel->linkinv(eta);
        vec devtemp = mModel->devResids(mY,mu,mWeight);
        mDev = sum(devtemp);
        mat mueta = mModel->muEta(eta);
        mat w = sqrt((mWeight * (mueta % mueta))/mModel->variance(mu)); //^2符号的作用？
        mResiduals = (mY - mu)/mueta;
    }
    else{
        eta = mModel->linkfun(mMuStart);
        mu = mModel->linkinv(eta);
        vec devoldtemp = mModel->devResids(mY,mu,mWeight);
        double devold = sum(devoldtemp);
        mat start = mat(nVars, mX.n_rows, fill::zeros);
        mat coef;
        for(int iter = 0; iter < mMaxit ; iter++){
            mat varmu = mModel->variance(mu);
            mat muetaval = mModel->muEta(eta);
            mat z = (eta - mOffset) + (mY - mu)/muetaval;
            mat w = sqrt((mWeight % (muetaval % muetaval))/varmu);
            mat xadj = mat(mX.n_rows,mX.n_cols);
            for (uword i = 0; i < mX.n_cols ; i++){
                xadj.col(i) = mX.col(i)%w;
            }
            for (uword i = 0; i < mX.n_rows ; i++){
                start.col(i) = CGwmGGWR::gwReg(xadj, z%w, vec(mX.n_rows,fill::ones));
            }
            eta = CGwmGGWR::Fitted(mX , start.t()) ; //?不是很确定
            mu = mModel->linkinv(eta + mOffset);
            mDev = sum(mModel->devResids(mY,mu,mWeight));
            if (isinf(mDev) ) {
                int ii = 1;
                while (isinf(mDev) ) {
                    if (ii > mMaxit){
                        return;
                    }
                    ii++;
                    start = (start + coefold)/2;
                    eta = CGwmGGWR::Fitted(mX , start.t());
                    mDev = sum(mModel->devResids(mY,mu,mWeight));
                }
            }
            if (abs(mDev - devold)/(0.1 + abs(mDev)) < mEpsilon ) {
                       //conv = true;
                       coef = start;
                       break;
                   } else {
                       devold = mDev;
                       coef = coefold = start;
            }
        }
    }
    vec wtdmu = (mIntercept)? sum(mWeight % mY)/sum(mWeight) : mModel->linkinv(mOffset);
    vec wtdmu1 = vec(mY.n_rows);
    for(uword i = 0; i < wtdmu1.n_rows ; i++){
        wtdmu1.row(i) = wtdmu;
    }
    vec nulldevtemp = mModel->devResids(mY,wtdmu1,mWeight);
    mNullDev = sum(nulldevtemp);
    uword rank = empty? 0 : mX.n_rows;
    mAIC = mModel->aic(mY,n,mu,mWeight) + 2.0 * rank;
}


bool CGwmGeneralizedLinearModel::setX(mat X){
    mX = X;
    return true;
}

bool CGwmGeneralizedLinearModel::setY(mat Y){
    mY =  Y;
    return true;
}

bool CGwmGeneralizedLinearModel::setFamily(CGwmGGWR::Family family){
    mFamily = family;
    return true;
}

double CGwmGeneralizedLinearModel::aic(){
    return mAIC;
}

double CGwmGeneralizedLinearModel::nullDev(){
    return mNullDev;
}

double CGwmGeneralizedLinearModel::dev(){
    return mDev;
}

bool CGwmGeneralizedLinearModel::checkCanceled()
{
    if(isCanceled())
    {
        return true;
    }
    else
    {
        return false;
    }
}

