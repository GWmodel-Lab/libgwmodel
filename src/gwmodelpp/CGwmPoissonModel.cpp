#include "CGwmPoissonModel.h"
//#include "GWmodel.h"
#include "CGwmGWRGeneralized.h"

using namespace arma;

CGwmPoissonModel::CGwmPoissonModel()
{

}

mat CGwmPoissonModel::initialize(){
    mat n = ones(mY.n_rows);
    mMuStart = mY + 0.1;
    return n;
}

mat CGwmPoissonModel::muStart(){
    return mMuStart;
}


mat CGwmPoissonModel::linkinv(mat eta){
    mat res = exp(eta);
    double eps = 2.220446e-16;
    if(res.min() > eps){
        return res;
    }
    for(int i = 0; (uword)i < eta.n_rows; i++){
        res(i) = res(i) > eps? res(i) : eps;
    }
    return res;
}

mat CGwmPoissonModel::variance(mat mu){
    return mu;
}

vec CGwmPoissonModel::devResids(mat y, mat mu, mat weights){
    mat r = mu % weights;  
    if( y.min() > 0){
        r = (weights % (y % log(y/mu) - (y - mu)));
    }
    else{
        for(int i = 0; (uword)i < y.n_rows; i++){
            if(y(i) > 0){
                r(i) = (weights(i) * (y(i) * log(y(i)/mu(i)) - (y(i) - mu(i))));
            }
        }
    }
    return 2 * r;
}

double CGwmPoissonModel::aic(mat y, mat n, mat mu, mat wt) {
    vec temp = CGwmGWRGeneralized::dpois(y, mu) % wt;
    return -2 * sum(temp);
}

mat CGwmPoissonModel::muEta(mat eta){
    mat res = exp(eta);
    double eps = 2.220446e-16;
    if(res.min() > eps){
        return res;
    }
    for(int i = 0; (uword)i < eta.n_rows; i++){
        res(i) = res(i) > eps? res(i) : eps;
    }
    return res;
}

bool CGwmPoissonModel::valideta(mat eta){
    return true;
}

bool CGwmPoissonModel::validmu(mat mu){
    if(mu.is_finite() && mu.min() > 0){
        return true;
    }
    return false;
}

mat CGwmPoissonModel::linkfun(mat muStart){
    mat res = log(muStart);
    return res;
}

bool CGwmPoissonModel::setY(mat y){
    mY = y;
    return true;
}

bool CGwmPoissonModel::setMuStart(mat muStart){
    mMuStart = muStart;
    return true;
}

bool CGwmPoissonModel::setWeight(mat weight){
    mWeight = weight;
    return true;
}

mat CGwmPoissonModel::weights(){
    return mWeight;
}

mat CGwmPoissonModel::getY(){
    return mY;
}


