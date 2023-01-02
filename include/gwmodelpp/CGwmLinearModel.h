#ifndef GWMLINEARMODEL_H
#define GWMLINEARMODEL_H

#include <armadillo>
using namespace arma;

class CGwmLinearModel
{
public:
    virtual mat initialize() = 0;
    virtual mat variance(mat mu) = 0;
    virtual mat linkinv(mat eta) = 0;
    virtual vec devResids(mat y,mat mu,mat weights) = 0;
    virtual double aic(mat y,mat n,mat mu,mat wt,double dev) = 0;
    virtual mat muEta(mat eta) = 0;
    virtual bool valideta(mat eta) = 0;
    virtual bool validmu(mat mu) = 0;
    virtual mat linkfun(mat muStart) = 0;

    virtual mat muStart() = 0;
    virtual mat weights() = 0;
    virtual mat getY() = 0;

    virtual bool setMuStart(mat muStart) = 0;
    virtual bool setY(mat y) = 0;
    virtual bool setWeight(mat weight) = 0;
};

#endif // GWMLINEARMODEL_H
