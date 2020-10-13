#include "gwmodelpp/spatialweight/CGwmDMatDistance.h"
#include <assert.h>

CGwmDMatDistance::CGwmDMatDistance(string dmatFile) : CGwmDistance()
{
    mDMatFile = dmatFile;
}

CGwmDMatDistance::CGwmDMatDistance(const CGwmDMatDistance &distance) : CGwmDistance(distance)
{
    mDMatFile = distance.mDMatFile;
}

vec CGwmDMatDistance::distance(DistanceParameter* parameter, uword focus)
{
    assert(parameter != nullptr);
    // QFile dmat(mDMatFile);
    // if (focus < mTotal && dmat.open(QFile::QIODevice::ReadOnly))
    // {
    //     qint64 basePos = 2 * sizeof (int);
    //     dmat.seek(basePos + focus * mRowSize * sizeof (double));
    //     QByteArray values = dmat.read(mRowSize * sizeof (double));
    //     return vec((double*)values.data(), mRowSize);
    // }
    return vec();
}
