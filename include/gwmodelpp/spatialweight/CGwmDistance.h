#ifndef CGWMDISTANCE_H
#define CGWMDISTANCE_H

#include <unordered_map>
#include <string>
#include <armadillo>

using namespace std;
using namespace arma;

class CGwmDistance
{
public:
    enum DistanceType
    {
        CRSDistance,
        MinkwoskiDistance,
        DMatDistance
    };
    
    static unordered_map<DistanceType, string> TypeNameMapper;

public:
    explicit CGwmDistance(int total) : mTotal(total) {};
    CGwmDistance(const CGwmDistance& d) { mTotal = d.mTotal; };
    virtual ~CGwmDistance() {};

    virtual CGwmDistance* clone() = 0;

    virtual DistanceType type() = 0;

    uword total() const;
    void setTotal(int total);


public:
    virtual vec distance(int focus) = 0;
    virtual uword length() const = 0;

    double maxDistance();
    double minDistance();

protected:
    uword mTotal = 0;
};

inline uword CGwmDistance::total() const
{
    return mTotal;
}

inline void CGwmDistance::setTotal(int total)
{
    mTotal = total;
}


#endif // CGWMDISTANCE_H
