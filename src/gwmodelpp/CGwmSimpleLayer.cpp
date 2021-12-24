#include "CGwmSimpleLayer.h"
#include <assert.h>

CGwmSimpleLayer::CGwmSimpleLayer(const mat& points, const mat& data, const vector<string>& fields)
{
    assert(points.n_cols == 2);
    assert(points.n_rows == data.n_rows);
    assert(data.n_cols == fields.size());
    mPoints = points;
    mData = data;
    mFields = fields;
}

CGwmSimpleLayer::CGwmSimpleLayer(const CGwmSimpleLayer& layer)
{
    mPoints = layer.mPoints;
    mData = layer.mData;
    mFields = layer.mFields;
}

CGwmSimpleLayer::~CGwmSimpleLayer()
{
}

