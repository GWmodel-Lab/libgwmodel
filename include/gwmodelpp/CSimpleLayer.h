#ifndef CSIMPLELAYER_H
#define CSIMPLELAYER_H

#include <armadillo>
using namespace arma;

namespace gwmodel
{

class CSimpleLayer
{
public:
    CSimpleLayer(const mat& points, const mat& data);
    virtual ~CSimpleLayer();

    mat points() const;
    mat data() const;

private:
    mat mPoints;
    mat mData;
};

inline mat CSimpleLayer::points() const
{
    return mPoints;
}

inline mat CSimpleLayer::data() const
{
    return mData;
}

} // namespace gwmodel


#endif  // CSIMPLELAYER_H