#include "gwmodelpp/spatialweight/DMatDistance.h"
#include <assert.h>

using namespace arma;
using namespace std;
using namespace gwm;

DMatDistance::DMatDistance(string dmatFile)
{
    mDMatFile = dmatFile;
}

DMatDistance::DMatDistance(const DMatDistance &distance)
{
    mDMatFile = distance.mDMatFile;
}

void DMatDistance::makeParameter(initializer_list<DistParamVariant> plist)
{
    if (plist.size() == 2)
    {
        const uword size = get<uword>(*(plist.begin()));
        const uword rows = get<uword>(*(plist.begin() + 1));
        mParameter = make_unique<Parameter>(size, rows);
    }
    else mParameter = nullptr;
}

vec DMatDistance::distance(uword focus)
{
    if (mParameter == nullptr) throw std::runtime_error("Parameter is nullptr.");
    if (focus >= mParameter->total) throw std::runtime_error("Index exceeds ranges.");
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

double DMatDistance::maxDistance()
{
    if(mParameter == nullptr) throw std::runtime_error("Parameter is nullptr.");
    return DBL_MAX;
}

double DMatDistance::minDistance()
{
    if(mParameter == nullptr) throw std::runtime_error("Parameter is nullptr.");
    return 0.0;
}
