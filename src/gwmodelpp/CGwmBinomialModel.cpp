#include "CGwmBinomialModel.h"
//#include "GWmodel.h"
#include "CGwmGGWR.h"

CGwmBinomialModel::CGwmBinomialModel()
{

}

mat CGwmBinomialModel::initialize(){
    int nCol = mY.n_cols;
    mat n;
    if(nCol == 1){
        n = ones(mY.n_rows);
        mMuStart = (mWeight % mY + 0.5)/(mWeight + 1);
    }
    else if(nCol == 2){
        mat y1 = mY.col(0);
        n = y1 + mY.col(1);
        mY = y1 / n;
        mWeight = mWeight % n;
        mMuStart = (n % mY + 0.5) / (n + 1);
    }
    return n;
}

mat CGwmBinomialModel::muStart(){
    return mMuStart;
}


mat CGwmBinomialModel::linkinv(mat eta){
    mat temp = exp(eta);
    mat res = temp / (1 + temp);
    return res;
}

mat CGwmBinomialModel::variance(mat mu){
    return mu % (1 - mu);
}

vec CGwmBinomialModel::devResids(mat y, mat mu, mat weights){
    int n = y.n_rows;
    mat res = vec(n);
    int lmu = mu.n_elem;
    if( lmu > 1){
        for (int i = 0;i < n; i++){
            double mui = mu[i];
            double yi = y[i];
            res[i] = 2 * weights[i] * (y_log_y(yi, mui) + y_log_y(1 - yi, 1 - mui));
        }
    }
    else{
        double mui = mu[0];
        for (int i = 0;i < n; i++){
            double yi = y[i];
            res[i] = 2 * weights[i] * (y_log_y(yi, mui) + y_log_y(1 - yi, 1 - mui));
        }
    }
    return res;
}

double CGwmBinomialModel::aic(mat y, mat n, mat mu, mat wt, double dev){
    mat m;
    if(n.max() > 1){
        m = n;
    }
    else
        m = wt;
    mat wi = vec(m.n_rows);
    for(int i = 0;(uword)i < m.n_rows; i++){
        if(m[i]>0){
            wi[i] = wt[i]/m[i];
        }
        else{
            wi[i] = 0;
        }
    }
    vec temp = wi % CGwmGGWR::dbinom(round(m % y), round(m), mu);
    return -2 * sum(temp);
}

mat CGwmBinomialModel::muEta(mat eta){
    mat temp = exp(eta);
    mat res = temp / ((1 + temp) % (1 + temp));
    return res;
}

bool CGwmBinomialModel::valideta(mat eta){
    return true;
}

bool CGwmBinomialModel::validmu(mat mu){
    if(mu.is_finite() && mu.min() > 0 && mu.max() < 1){
        return true;
    }
    return false;
}

mat CGwmBinomialModel::linkfun(mat muStart){
    mat temp = muStart /(1 - muStart);
    mat res = log(temp);
    return res;
}

bool CGwmBinomialModel::setY(mat y){
    mY = y;
    return true;
}

bool CGwmBinomialModel::setMuStart(mat muStart){
    mMuStart = muStart;
    return true;
}

bool CGwmBinomialModel::setWeight(mat weight){
    mWeight = weight;
    return true;
}

mat CGwmBinomialModel::weights(){
    return mWeight;
}

mat CGwmBinomialModel::getY(){
    return mY;
}

double CGwmBinomialModel::y_log_y(double y, double mu)
{
    return (y != 0.) ? (y * log(y/mu)) : 0;
}
