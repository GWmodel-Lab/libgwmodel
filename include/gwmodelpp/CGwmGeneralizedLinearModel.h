#ifndef GWMGENERALIZEDLINEARMODEL_H
#define GWMGENERALIZEDLINEARMODEL_H

#include <armadillo>
#include "CGwmLinearModel.h"
#include "CGwmGGWR.h"

using namespace arma;

class CGwmGeneralizedLinearModel
{
public:
    CGwmGeneralizedLinearModel();

protected:
    mat mX;
    mat mY;
    mat mWeight;
    CGwmGGWR::Family mFamily;
    double mEpsilon;
    int mMaxit;
    bool mIntercept;
    mat mOffset;
    CGwmLinearModel* mModel;

    mat mMuStart;
    double mDev;
    mat mResiduals;
    double mNullDev;
    double mAIC;
    bool mIsCanceled = false;

public:
    void fit();

    bool setX(mat X);
    bool setY(mat Y);
    bool setFamily(CGwmGGWR::Family family);

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
