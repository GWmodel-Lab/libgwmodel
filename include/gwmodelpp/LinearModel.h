#ifndef GWMLINEARMODEL_H
#define GWMLINEARMODEL_H

#include <armadillo>

namespace gwm
{

class LinearModel
{
public:
    virtual arma::mat initialize() = 0;
    virtual arma::mat variance(arma::mat mu) = 0;
    virtual arma::mat linkinv(arma::mat eta) = 0;
    virtual arma::vec devResids(arma::mat y,arma::mat mu,arma::mat weights) = 0;
    virtual double aic(arma::mat y,arma::mat n,arma::mat mu,arma::mat wt) = 0;
    virtual arma::mat muEta(arma::mat eta) = 0;
    virtual bool valideta(arma::mat eta) = 0;
    virtual bool validmu(arma::mat mu) = 0;
    virtual arma::mat linkfun(arma::mat muStart) = 0;

    virtual arma::mat muStart() = 0;
    virtual arma::mat weights() = 0;
    virtual arma::mat getY() = 0;

    virtual bool setMuStart(arma::mat muStart) = 0;
    virtual bool setY(arma::mat y) = 0;
    virtual bool setWeight(arma::mat weight) = 0;
};

}

#endif // GWMLINEARMODEL_H
