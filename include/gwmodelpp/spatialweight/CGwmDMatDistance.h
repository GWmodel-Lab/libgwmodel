#ifndef CGWMDMATDISTANCE_H
#define CGWMDMATDISTANCE_H

#include <string>
#include "CGwmDistance.h"

using namespace std;

struct DMatDistanceParameter : public DistanceParameter
{
    int rowSize;

    DMatDistanceParameter(int size, uword rows) : rowSize(size) 
    {
        total = rows;
    }
};

/**
 * [NOT AVALIABLE]
 */
class CGwmDMatDistance : public CGwmDistance
{
public:
    explicit CGwmDMatDistance(string dmatFile);
    CGwmDMatDistance(const CGwmDMatDistance& distance);

    virtual CGwmDistance * clone() override
    {
        return new CGwmDMatDistance(*this);
    }

    DistanceType type() override { return DistanceType::DMatDistance; }

    string dMatFile() const;
    void setDMatFile(const string &dMatFile);

public:
    virtual vec distance(DistanceParameter* parameter, uword focus) override;

private:
    string mDMatFile;
};

inline string CGwmDMatDistance::dMatFile() const
{
    return mDMatFile;
}

inline void CGwmDMatDistance::setDMatFile(const string &dMatFile)
{
    mDMatFile = dMatFile;
}

#endif // CGWMDMATDISTANCE_H
