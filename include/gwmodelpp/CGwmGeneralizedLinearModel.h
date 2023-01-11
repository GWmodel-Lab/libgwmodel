#ifndef GWMGENERALIZEDLINEARMODEL_H
#define GWMGENERALIZEDLINEARMODEL_H

#include <armadillo>
#include "CGwmLinearModel.h"
#include "CGwmGWRGeneralized.h"


class CGwmGeneralizedLinearModel
{
public:
    CGwmGeneralizedLinearModel();

protected:
    arma::mat mX;
    arma::mat mY;
    arma::mat mWeight;
    CGwmGWRGeneralized::Family mFamily;
    double mEpsilon;
    int mMaxit;
    bool mIntercept;
    arma::mat mOffset;
    CGwmLinearModel* mModel;

    arma::mat mMuStart;
    double mDev;
    arma::mat mResiduals;
    double mNullDev;
    double mAIC;
    bool mIsCanceled = false;

public:
    void fit();

    bool setX(arma::mat X);
    bool setY(arma::mat Y);
    bool setFamily(CGwmGWRGeneralized::Family family);

    double dev();
    double nullDev();
    double aic();
    bool isCanceled() const;
    void setCanceled(bool newCanceled);
    bool checkCanceled();


};
inline bool CGwmGeneralizedLinearModel::isCanceled() const
{
    return mIsCanceled;
}

inline void CGwmGeneralizedLinearModel::setCanceled(bool newCanceled)
{
    mIsCanceled = newCanceled;
}

#endif // GWMGENERALIZEDLINEARMODEL_H
