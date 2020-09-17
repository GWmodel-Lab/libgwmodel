#ifndef GWMVARIABLE_H
#define GWMVARIABLE_H

#include <string>
#include "gwmodelpp.h"

struct GWMODELPP_API GwmVariable
{
    int index;
    bool isNumeric;
    std::string name;
};

#endif  // GWMVARIABLE_H