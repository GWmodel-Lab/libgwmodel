#include "CGwmSimpleLayer.h"

CGwmSimpleLayer::CGwmSimpleLayer(const mat& points, const mat& data)
{
    _ASSERT(points.n_cols == 2);
    _ASSERT(points.n_rows == data.n_rows);
    mPoints = points;
    mData = data;
}

CGwmSimpleLayer::~CGwmSimpleLayer()
{
}

