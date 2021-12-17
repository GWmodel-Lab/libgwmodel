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

    /**
     * @brief Construct a new CGwmGWPCA object.
     * 
     * Use gwmodel_create_gwpca_algorithm() to construct an instance in shared build.
     */
    CGwmGWPCA();
    
    virtual ~CGwmGWPCA();

    int keepComponents();
    void setKeepComponents(int k);

    mat localPV();
    cube loadings();
    mat sdev();
    cube scores();

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

private:    // Algorithm Parameters
    vector<GwmVariable> mVariables;
    int mK = 2;
    // bool mRobust = false;

private:    // Algorithm Results
    mat mLocalPV;
    cube mLoadings;
    mat mSDev;
    cube mScores;
    vector<string> mWinner;

private:    // Algorithm Runtime Variables
    mat mX;
    vec mLatestWt;

    Solver mSolver = &CGwmGWPCA::solveSerial;
    DistanceParameter* mDistanceParameter = nullptr;
};

inline int CGwmGWPCA::keepComponents()
{
    return mK;
}

inline void CGwmGWPCA::setKeepComponents(int k)
{
    mK = k;
}

inline mat CGwmGWPCA::localPV()
{
    return mLocalPV;
}

inline cube CGwmGWPCA::loadings()
{
    return mLoadings;
}

inline mat CGwmGWPCA::sdev()
{
    return mSDev;
}

inline cube CGwmGWPCA::scores()
{
    return mScores;
}

inline vector<GwmVariable> CGwmGWPCA::variables() const
{
    return mVariables;
}

inline void CGwmGWPCA::setVariables(const vector<GwmVariable>& variables)
{
    mVariables = variables;
}

#endif  // CGWMGWPCA_H