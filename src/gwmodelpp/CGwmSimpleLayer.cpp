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

CGwmSimpleLayer::~CGwmSimpleLayer()
{
}

