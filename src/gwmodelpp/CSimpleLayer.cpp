#include "CSimpleLayer.h"

using namespace gwmodel;

CSimpleLayer::CSimpleLayer(const mat& points, const mat& data)
{
    mPoints = points;
    mData = data;
}

CSimpleLayer::~CSimpleLayer()
{
}

