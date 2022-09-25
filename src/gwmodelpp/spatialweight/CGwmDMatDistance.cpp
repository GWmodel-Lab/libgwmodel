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

void CGwmDMatDistance::makeParameter(initializer_list<DistParamVariant> plist)
{
    if (plist.size() == 2)
    {
        const uword size = get<uword>(*(plist.begin()));
        const uword rows = get<uword>(*(plist.begin() + 1));
        mParameter = make_unique<Parameter>(size, rows);
    }
    else mParameter = nullptr;
}

vec CGwmDMatDistance::distance(uword focus)
{
    if(mParameter == nullptr) throw std::runtime_error("Parameter is nullptr.");
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

double CGwmDMatDistance::maxDistance()
{
    if(mParameter == nullptr) throw std::runtime_error("Parameter is nullptr.");
    return DBL_MAX;
}

double CGwmDMatDistance::minDistance()
{
    if(mParameter == nullptr) throw std::runtime_error("Parameter is nullptr.");
    return 0.0;
}
