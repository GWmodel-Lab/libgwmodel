#ifndef CGWMDMATDISTANCE_H
#define CGWMDMATDISTANCE_H

#include <string>
#include "spatialweight/CGwmDistance.h"

using namespace std;

class CGwmDMatDistance : public CGwmDistance
{
public:
    explicit CGwmDMatDistance(int total, string dmatFile);
    CGwmDMatDistance(const CGwmDMatDistance& distance);

    virtual CGwmDistance * clone() override
    {
        return new CGwmDMatDistance(*this);
    }

    DistanceType type() override { return DistanceType::DMatDistance; }

    string dMatFile() const;
    void setDMatFile(const string &dMatFile);

public:
    virtual vec distance(int focus) override;
    uword length() const override;

    int rowSize() const;
    void setRowSize(int rowSize);

private:
    string mDMatFile;

    int mRowSize = 0;
};

inline string CGwmDMatDistance::dMatFile() const
{
    return mDMatFile;
}

inline void CGwmDMatDistance::setDMatFile(const string &dMatFile)
{
    mDMatFile = dMatFile;
}

inline uword CGwmDMatDistance::length() const
{
    return rowSize();
}

inline int CGwmDMatDistance::rowSize() const
{
    return mRowSize;
}

inline void CGwmDMatDistance::setRowSize(int rowSize)
{
    mRowSize = rowSize;
}

#endif // CGWMDMATDISTANCE_H
