#include "PoissonModel.h"
//#include "GWmodel.h"
#include "GWRGeneralized.h"

using namespace arma;
using namespace gwm;

PoissonModel::PoissonModel()
{

}

mat PoissonModel::initialize(){
    mat n = ones(mY.n_rows);
    mMuStart = mY + 0.1;
    return n;
}

mat PoissonModel::muStart(){
    return mMuStart;
}


mat PoissonModel::linkinv(mat eta){
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

mat PoissonModel::variance(mat mu){
    return mu;
}

vec PoissonModel::devResids(mat y, mat mu, mat weights){
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

double PoissonModel::aic(mat y, mat n, mat mu, mat wt) {
    vec temp = GWRGeneralized::dpois(y, mu) % wt;
    return -2 * sum(temp);
}

mat PoissonModel::muEta(mat eta){
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

bool PoissonModel::valideta(mat eta){
    return true;
}

bool PoissonModel::validmu(mat mu){
    if(mu.is_finite() && mu.min() > 0){
        return true;
    }
    return false;
}

mat PoissonModel::linkfun(mat muStart){
    mat res = log(muStart);
    return res;
}

bool PoissonModel::setY(mat y){
    mY = y;
    return true;
}

bool PoissonModel::setMuStart(mat muStart){
    mMuStart = muStart;
    return true;
}

bool PoissonModel::setWeight(mat weight){
    mWeight = weight;
    return true;
}

mat PoissonModel::weights(){
    return mWeight;
}

mat PoissonModel::getY(){
    return mY;
}


