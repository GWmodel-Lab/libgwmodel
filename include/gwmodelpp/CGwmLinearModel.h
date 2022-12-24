#ifndef GWMLINEARMODEL_H
#define GWMLINEARMODEL_H

#include <armadillo>
using namespace arma;

class CGwmLinearModel
{
public:
    CGwmLinearModel();
public:
    virtual mat initialize() {return nullptr;};
    virtual mat variance(mat mu) {return nullptr;};
    virtual mat linkinv(mat eta) {return nullptr;};
    virtual mat devResids(mat y,mat mu,mat weights) {return nullptr;};
    virtual double aic(mat y,mat n,mat mu,mat wt,double dev) {return 0.0;};
    virtual mat muEta(mat eta) {return nullptr;};
    virtual bool valideta(mat eta) {return true;};
    virtual bool validmu(mat mu) {return true;};
    virtual mat linkfun(mat muStart) {return nullptr;};

    virtual mat muStart() {return nullptr;};
    virtual mat weights() {return nullptr;};
    virtual mat getY() {return nullptr;};

    virtual bool setMuStart(mat muStart) {return true;};
    virtual bool setY(mat y) {return true;};
    virtual bool setWeight(mat weight) {return true;};
};

#endif // GWMLINEARMODEL_H
