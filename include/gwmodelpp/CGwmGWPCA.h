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
private:
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
    
    /**
     * @brief Destroy the CGwmGWPCA object.
     * 
     * Use gwmodel_delete_gwpca_algorithm() to destory an instance in shared build.
     */
    virtual ~CGwmGWPCA();

    /**
     * @brief Get the number of Kept Components.
     * 
     * @return int Number of Kept Components.
     */
    int keepComponents();

    /**
     * @brief Set the number of Kept Components object.
     * 
     * @param k Number of Kept Components.
     */
    void setKeepComponents(int k);

    /**
     * @brief Get the Local Principle Values matrix.
     * 
     * @return mat Local Principle Values matrix.
     */
    mat localPV();

    /**
     * @brief Get the Loadings matrix.
     * 
     * @return mat Loadings matrix.
     */
    cube loadings();

    /**
     * @brief Get the Standard deviation matrix.
     * 
     * @return mat Standard deviation matrix.
     */
    mat sdev();

    /**
     * @brief Get the Scores matrix.
     * 
     * @return mat Scores matrix.
     */
    cube scores();

public: // IGwmMultivariableAnalysis
    virtual vector<GwmVariable> variables() const;
    virtual void setVariables(const vector<GwmVariable>& variables);

public: // GwmAlgorithm
    virtual void run();
    virtual bool isValid();

private:

    /**
     * @brief Create a Result Layer object.
     * 
     * @param items Result Layer objects.
     */
    void createResultLayer(vector<ResultLayerDataItem> items);

private:

    /**
     * @brief Set CGwmGWPCA::mX according to layer and variables. 
     * If there are \f$n\f$ features in layer and \f$k\f$ elements in variables, this function will set matrix x to the shape \f$ n \times k \f$.
     * Its element in location \f$ (i,j) \f$ will equal to the value of i-th feature's j-th variable. 
     * 
     * @param x Reference of CGwmGWSS::mX or other matrix to store value of variables.
     * @param layer Pointer to source data layer. 
     * @param variables Vector of variables.
     */
    void setX(mat& x, const CGwmSimpleLayer* layer, const vector<GwmVariable>& variables);

    /**
     * @brief Create a Distance Parameter object. Store in CGwmGWSS::mDistanceParameter.
     */
    void createDistanceParameter();

    /**
     * @brief Function to carry out PCA.
     * 
     * @param x Symmetric data matrix.
     * @param loadings Out reference to loadings matrix.
     * @param sdev Out reference to standard deviation matrix.
     * @return mat Principle values matrix.
     */
    mat pca(const mat& x, cube& loadings, mat& sdev)
    {
        return (this->*mSolver)(x, loadings, sdev);
    }

    /**
     * @brief Serial version of PCA funtion.
     * 
     * @param x Symmetric data matrix.
     * @param loadings Out reference to loadings matrix.
     * @param sdev Out reference to standard deviation matrix.
     * @return mat Principle values matrix.
     */
    mat solveSerial(const mat& x, cube& loadings, mat& sdev);

    /**
     * @brief Function to carry out weighted PCA.
     * 
     * @param x Symmetric data matrix.
     * @param w Weight vector.
     * @param V Right orthogonal matrix.
     * @param d Rectangular diagonal matrix
     */
    void wpca(const mat& x, const vec& w, mat& V, vec & d);

private:    // Algorithm Parameters
    vector<GwmVariable> mVariables;
    int mK = 2;  //< Number of components to be kept.
    // bool mRobust = false;

private:    // Algorithm Results
    mat mLocalPV;               //< Local principle component values.
    cube mLoadings;             //< Loadings for each component.
    mat mSDev;                  //< Standard Deviation.
    cube mScores;               //< Scores for each variable.
    vector<string> mWinner;     //< Winner variable at each sample.

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