#ifndef CGWMGWPCA_H
#define CGWMGWPCA_H

#include <armadillo>
#include <vector>
#include "CGwmSpatialMonoscaleAlgorithm.h"
#include "IGwmMultivariableAnalysis.h"
#include "IGwmParallelizable.h"

using namespace std;

class CGwmGWPCA: public CGwmSpatialMonoscaleAlgorithm, public IGwmMultivariableAnalysis
{

public: // Constructors and Deconstructors
    CGwmGWPCA(/* args */) {}
    ~CGwmGWPCA() {}

public: // IGwmMultivariableAnalysis
    virtual vector<GwmVariable> variables() const;
    virtual void setVariables(const vector<GwmVariable>& variables);

public:
    virtual void run();
    virtual bool isValid();

private:
    void setX(mat& x, const CGwmSimpleLayer* layer, const vector<GwmVariable>& variables);

    /**
     * @brief Create a Distance Parameter object. Store in CGwmGWSS::mDistanceParameter.
     */
    void createDistanceParameter();

    void solveSerial();

private:
    vector<GwmVariable> mVariables;

    mat mX;
    vec mLatestWt;
    int mK = 2;
    bool mRobust = false;
};

#endif  // CGWMGWPCA_H