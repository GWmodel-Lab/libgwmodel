#ifndef GWMBINOMIALMODEL_H
#define GWMBINOMIALMODEL_H

#include "CGwmLinearModel.h"

class CGwmBinomialModel : public CGwmLinearModel
{
public:
    CGwmBinomialModel();

public:
    mat mMuStart;
    mat mY;
    mat mWeight;

public:
    mat initialize() override;
    mat variance(mat mu) override;
    mat linkinv(mat eta) override;
    vec devResids(mat y,mat mu,mat weights) override;
    double aic(mat y,mat n,mat mu,mat wt) override;
    mat muEta(mat eta) override;
    bool valideta(mat eta) override;
    bool validmu(mat mu) override;
    mat linkfun(mat muStart) override;

    mat muStart() override;
    mat weights() override;
    mat getY() override;
    bool setMuStart(mat muStart) override;
    bool setY(mat y) override;
    bool setWeight(mat weight) override;

protected:
    double y_log_y(double y, double mu);
};

#endif // GWMBINOMIALMODEL_H
