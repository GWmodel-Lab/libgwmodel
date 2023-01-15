#ifndef REGRESSIONDIAGNOSTIC_H
#define REGRESSIONDIAGNOSTIC_H

namespace gwm
{

struct RegressionDiagnostic
{
    double RSS;
    double AIC;
    double AICc;
    double ENP;
    double EDF;
    double RSquare;
    double RSquareAdjust;
};

}

#endif  // REGRESSIONDIAGNOSTIC_H