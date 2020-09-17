#ifndef GWMREGRESSIONDIAGNOSTIC_H
#define GWMREGRESSIONDIAGNOSTIC_H
#include "gwmodelpp.h"

struct GWMODELPP_API GwmRegressionDiagnostic
{
    double RSS;
    double AIC;
    double AICc;
    double ENP;
    double EDF;
    double RSquare;
    double RSquareAdjust;
};

#endif  // GWMREGRESSIONDIAGNOSTIC_H