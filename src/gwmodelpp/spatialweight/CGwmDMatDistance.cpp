#include "spatialweight/CGwmDMatDistance.h"

CGwmDMatDistance::CGwmDMatDistance(int total, string dmatFile) : CGwmDistance(total)
{
    mDMatFile = dmatFile;
    mRowSize = total;
}

CGwmDMatDistance::CGwmDMatDistance(const CGwmDMatDistance &distance) : CGwmDistance(distance)
{
    mDMatFile = distance.mDMatFile;
    mRowSize = distance.mRowSize;
}

vec CGwmDMatDistance::distance(int focus)
{
    // QFile dmat(mDMatFile);
    // if (focus < mTotal && dmat.open(QFile::QIODevice::ReadOnly))
    // {
    //     qint64 basePos = 2 * sizeof (int);
    //     dmat.seek(basePos + focus * mRowSize * sizeof (double));
    //     QByteArray values = dmat.read(mRowSize * sizeof (double));
    //     return vec((double*)values.data(), mRowSize);
    // }
    return vec(mRowSize, fill::zeros);
}
