#ifndef GWMPOISSONMODEL_H
#define GWMPOISSONMODEL_H

#include "CGwmLinearModel.h"

class CGwmPoissonModel : public CGwmLinearModel
{
public:
    CGwmPoissonModel();

public:
    arma::mat mMuStart;
    arma::mat mY;
    arma::mat mWeight;

public:
    arma::mat initialize() override;
    arma::mat variance(arma::mat mu) override;
    arma::mat linkinv(arma::mat eta) override;
    arma::vec devResids(arma::mat y,arma::mat mu,arma::mat weights) override;
    double aic(arma::mat y,arma::mat n,arma::mat mu,arma::mat wt) override;
    arma::mat muEta(arma::mat eta) override;
    bool valideta(arma::mat eta) override;
    bool validmu(arma::mat mu) override;
    arma::mat linkfun(arma::mat muStart) override;

    arma::mat muStart() override;
    arma::mat weights() override;
    arma::mat getY() override;
    bool setMuStart(arma::mat muStart) override;
    bool setY(arma::mat y) override;
    bool setWeight(arma::mat weight) override;
};

#endif // GWMPOISSONMODEL_H
