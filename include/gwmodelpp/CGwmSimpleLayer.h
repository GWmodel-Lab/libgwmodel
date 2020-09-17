#ifndef CGWMSIMPLELAYER_H
#define CGWMSIMPLELAYER_H

#include <vector>
#include <string>
#include <armadillo>
#include "gwmodelpp.h"
using namespace std;
using namespace arma;

class GWMODELPP_API CGwmSimpleLayer
{
public:
    CGwmSimpleLayer(const mat& points, const mat& data, const vector<string>& fields);
    virtual ~CGwmSimpleLayer();

    mat points() const;
    mat data() const;
    vector<string> fields() const;

    uword featureCount() const;

private:
    mat mPoints;
    mat mData;
    vector<string> mFields;
};

inline mat CGwmSimpleLayer::points() const
{
    return mPoints;
}

inline mat CGwmSimpleLayer::data() const
{
    return mData;
}

inline vector<string> CGwmSimpleLayer::fields() const
{
    return mFields;
}

inline uword CGwmSimpleLayer::featureCount() const
{
    return mPoints.n_rows;
}


#endif  // CGWMSIMPLELAYER_H