#ifndef CGWMALGORITHM_H
#define CGWMALGORITHM_H

#include "gwmodelpp.h"

class GWMODELPP_API CGwmAlgorithm
{
public:
    CGwmAlgorithm();
    ~CGwmAlgorithm();

public:
    virtual void run() = 0;
};

#endif  // CGWMALGORITHM_H