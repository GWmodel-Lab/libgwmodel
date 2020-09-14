#ifndef CGWMSIMPLELAYER_H
#define CGWMSIMPLELAYER_H

#include <armadillo>
using namespace arma;

class CGwmSimpleLayer
{
public:
    CGwmSimpleLayer(const mat& points, const mat& data);
    virtual ~CGwmSimpleLayer();

    mat points() const;
    mat data() const;

private:
    mat mPoints;
    mat mData;
};

inline mat CGwmSimpleLayer::points() const
{
    return mPoints;
}

inline mat CGwmSimpleLayer::data() const
{
    return mData;
}


#endif  // CGWMSIMPLELAYER_H