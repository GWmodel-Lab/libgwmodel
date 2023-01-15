#ifndef IREGRESSIONANALYSIS_H
#define IREGRESSIONANALYSIS_H

#include <vector>
#include <armadillo>
#include "RegressionDiagnostic.h"


namespace gwm
{

struct IRegressionAnalysis
{
    virtual arma::vec dependentVariable() const = 0;
    virtual void setDependentVariable(const arma::vec& y) = 0;

    virtual arma::mat independentVariables() const = 0;
    virtual void setIndependentVariables(const arma::mat& x) = 0;
    
    virtual bool hasIntercept() const = 0;
    virtual void setHasIntercept(const bool has) = 0;

    virtual arma::mat predict(const arma::mat& locations) = 0;
    virtual arma::mat fit() = 0;

    virtual RegressionDiagnostic diagnostic() const = 0;
};

}

#endif  // IREGRESSIONANALYSIS_H