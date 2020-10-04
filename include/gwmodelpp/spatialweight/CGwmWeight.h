#ifndef CGWMWEIGHT_H
#define CGWMWEIGHT_H

#include <unordered_map>
#include <string>
#include <armadillo>

using namespace std;
using namespace arma;

class CGwmWeight
{
public:
    enum WeightType
    {
        BandwidthWeight
    };

    static unordered_map<WeightType, string> TypeNameMapper;

public:
    CGwmWeight() {}
    virtual ~CGwmWeight() {}

    virtual CGwmWeight* clone() = 0;

public:
    virtual vec weight(vec dist) = 0;
};

#endif // CGWMWEIGHT_H
