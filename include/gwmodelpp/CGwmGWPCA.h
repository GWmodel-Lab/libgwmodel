#ifndef CGWMGWPCA_H
#define CGWMGWPCA_H

#include <armadillo>
#include <vector>
#include <tuple>
#include "CGwmSpatialMonoscaleAlgorithm.h"
#include "IGwmMultivariableAnalysis.h"
#include "IGwmParallelizable.h"


class CGwmGWPCA: public CGwmSpatialMonoscaleAlgorithm, public IGwmMultivariableAnalysis
{
private:
    typedef mat (CGwmGWPCA::*Solver)(const mat&, cube&, mat&);

public: // Constructors and Deconstructors

    /**
     * @brief Construct a new CGwmGWPCA object.
     * 
     * Use gwmodel_create_gwpca_algorithm() to construct an instance in shared build.
     */
    CGwmGWPCA() {}

    /**
     * @brief Construct a new CGwmGWPCA object.
     * 
     * Use gwmodel_create_gwpca_algorithm() to construct an instance in shared build.
     */
    CGwmGWPCA(const arma::mat x, const arma::mat coords, const CGwmSpatialWeight& spatialWeight)
        : CGwmSpatialMonoscaleAlgorithm(spatialWeight, coords)
    {
        mX = x;
    }
    
    /**
     * @brief Destroy the CGwmGWPCA object.
     * 
     * Use gwmodel_delete_gwpca_algorithm() to destory an instance in shared build.
     */
    virtual ~CGwmGWPCA() {}

    /**
     * @brief Get the number of Kept Components.
     * 
     * @return int Number of Kept Components.
     */
    int keepComponents() { return mK; }

    /**
     * @brief Set the number of Kept Components object.
     * 
     * @param k Number of Kept Components.
     */
    void setKeepComponents(int k) { mK = k; }

    /**
     * @brief Get the Local Principle Values matrix.
     * 
     * @return mat Local Principle Values matrix.
     */
    mat localPV() { return mLocalPV; }

    /**
     * @brief Get the Loadings matrix.
     * 
     * @return mat Loadings matrix.
     */
    cube loadings() { return mLoadings; }

    /**
     * @brief Get the Standard deviation matrix.
     * 
     * @return mat Standard deviation matrix.
     */
    mat sdev() { return mSDev; }

    /**
     * @brief Get the Scores matrix.
     * 
     * @return mat Scores matrix.
     */
    cube scores() { return mScores; }

public: // IGwmMultivariableAnalysis
    virtual mat variables() const override { return mX; }
    virtual void setVariables(const mat& x) override { mX = x; }
    virtual void run() override;

public: // GwmAlgorithm
    virtual bool isValid() override;

private:

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
    int mK = 2;  //< Number of components to be kept.
    // bool mRobust = false;

private:    // Algorithm Results
    mat mLocalPV;               //< Local principle component values.
    cube mLoadings;             //< Loadings for each component.
    mat mSDev;                  //< Standard Deviation.
    cube mScores;               //< Scores for each variable.
    uvec mWinner;               //< Winner variable at each sample.

private:    // Algorithm Runtime Variables
    mat mX;
    vec mLatestWt;

    Solver mSolver = &CGwmGWPCA::solveSerial;
};

#endif  // CGWMGWPCA_H