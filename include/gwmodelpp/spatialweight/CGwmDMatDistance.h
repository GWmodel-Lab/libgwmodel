#ifndef CGWMDMATDISTANCE_H
#define CGWMDMATDISTANCE_H

#include <string>
#include "CGwmDistance.h"

using namespace std;

struct DMatDistanceParameter : public DistanceParameter
{
    uword rowSize;

    DMatDistanceParameter(uword size, uword rows) : rowSize(size) 
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

    /**
     * @brief Create Parameter for Caclulating CRS Distance.
     * 
     * @param plist A list of parameters containing 2 items:
     *  - `uword` size
     *  - `uword` rows
     *  . 
     * 
     * @return DistanceParameter* The pointer to parameters.
     */
    virtual DistanceParameter* makeParameter(initializer_list<DistParamVariant> plist) override;
    
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
