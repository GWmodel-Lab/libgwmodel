#ifndef CGWMDISTANCE_H
#define CGWMDISTANCE_H

#include <unordered_map>
#include <string>
#include <armadillo>
#include "gwmodelpp.h"

using namespace std;
using namespace arma;

struct GWMODELPP_API DistanceParameter
{
    uword focus;

    DistanceParameter(): focus(0) {}
};

class GWMODELPP_API CGwmDistance
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
    explicit CGwmDistance() {};
    CGwmDistance(const CGwmDistance& d) {};
    virtual ~CGwmDistance() {};

    virtual CGwmDistance* clone() = 0;

    virtual DistanceType type() = 0;


public:
    virtual vec distance(DistanceParameter* parameter) = 0;

    double maxDistance(uword total, DistanceParameter* parameter);
    double minDistance(uword total, DistanceParameter* parameter);
};


#endif // CGWMDISTANCE_H
