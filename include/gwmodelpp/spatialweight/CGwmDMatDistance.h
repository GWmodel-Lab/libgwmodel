#ifndef CGWMDMATDISTANCE_H
#define CGWMDMATDISTANCE_H

#include <string>
#include "gwmodelpp.h"
#include "spatialweight/CGwmDistance.h"

using namespace std;

struct GWMODELPP_API DMatDistanceParameter : public DistanceParameter
{
    int rowSize;

    DMatDistanceParameter() {}
};

/**
 * [NOT AVALIABLE]
 */
class GWMODELPP_API CGwmDMatDistance : public CGwmDistance
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
    virtual vec distance(DistanceParameter* parameter) override;

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
