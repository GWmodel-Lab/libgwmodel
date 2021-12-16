#ifndef CGWMGWPCA_H
#define CGWMGWPCA_H

#include <armadillo>
#include <vector>
#include <tuple>
#include "CGwmSpatialMonoscaleAlgorithm.h"
#include "IGwmMultivariableAnalysis.h"
#include "IGwmParallelizable.h"

using namespace std;

class CGwmGWPCA: public CGwmSpatialMonoscaleAlgorithm, public IGwmMultivariableAnalysis
{
    typedef mat (CGwmGWPCA::*Solver)(const mat&, cube&, mat&);

    enum NameFormat
    {
        Fixed,
        PrefixCompName
    };

    typedef tuple<string, mat, NameFormat> ResultLayerDataItem;

public: // Constructors and Deconstructors
    CGwmGWPCA();
    ~CGwmGWPCA();

public: // IGwmMultivariableAnalysis
    virtual vector<GwmVariable> variables() const;
    virtual void setVariables(const vector<GwmVariable>& variables);

public:
    virtual void run();
    virtual bool isValid();

private:
    void createResultLayer(vector<ResultLayerDataItem> items);

private:
    void setX(mat& x, const CGwmSimpleLayer* layer, const vector<GwmVariable>& variables);

    /**
     * @brief Create a Distance Parameter object. Store in CGwmGWSS::mDistanceParameter.
     */
    void createDistanceParameter();

    mat pca(const mat& x, cube& loadings, mat& sdev)
    {
        return (this->*mSolver)(x, loadings, sdev);
    }

    mat solveSerial(const mat& x, cube& loadings, mat& sdev);

    void wpca(const mat& x, const vec& w, mat& V, vec & d);

private:
    vector<GwmVariable> mVariables;

    mat mX;
    vec mLatestWt;
    int mK = 2;
    bool mRobust = false;

    mat mLocalPV;
    cube mLoadings;
    mat mSDev;
    cube mScores;
    vector<string> mWinner;

    Solver mSolver = &CGwmGWPCA::solveSerial;
    DistanceParameter* mDistanceParameter = nullptr;
};

#endif  // CGWMGWPCA_H