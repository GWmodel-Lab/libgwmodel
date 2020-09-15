#ifndef GWMREGRESSIONDIAGNOSTIC_H
#define GWMREGRESSIONDIAGNOSTIC_H

struct GwmRegressionDiagnostic
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