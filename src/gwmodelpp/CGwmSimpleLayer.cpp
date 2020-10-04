#include "CGwmSimpleLayer.h"

CGwmSimpleLayer::CGwmSimpleLayer(const mat& points, const mat& data, const vector<string>& fields)
{
    _ASSERT(points.n_cols == 2);
    _ASSERT(points.n_rows == data.n_rows);
    _ASSERT(data.n_cols == fields.size());
    mPoints = points;
    mData = data;
    mFields = fields;
}

CGwmSimpleLayer::~CGwmSimpleLayer()
{
}

