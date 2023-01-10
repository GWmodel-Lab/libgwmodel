#ifndef GENERALIZEDLINEARMODEL_H
#define GENERALIZEDLINEARMODEL_H

#include <armadillo>
#include "LinearModel.h"
#include "GWRGeneralized.h"


namespace gwm
{

class GeneralizedLinearModel
{
public:
    GeneralizedLinearModel();

protected:
    arma::mat mX;
    arma::mat mY;
    arma::mat mWeight;
    GWRGeneralized::Family mFamily;
    double mEpsilon;
    int mMaxit;
    bool mIntercept;
    arma::mat mOffset;
    LinearModel* mModel;

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
    bool setFamily(GWRGeneralized::Family family);

    double dev();
    double nullDev();
    double aic();
    bool isCanceled() const;
    void setCanceled(bool newCanceled);
    bool checkCanceled();


};
inline bool GeneralizedLinearModel::isCanceled() const
{
    return mIsCanceled;
}

inline void GeneralizedLinearModel::setCanceled(bool newCanceled)
{
    mIsCanceled = newCanceled;
}

}

#endif // GENERALIZEDLINEARMODEL_H
