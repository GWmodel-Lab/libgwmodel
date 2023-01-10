#ifndef GWPCA_H
#define GWPCA_H

#include <armadillo>
#include <vector>
#include <tuple>
#include "SpatialMonoscaleAlgorithm.h"
#include "IMultivariableAnalysis.h"
#include "IParallelizable.h"


namespace gwm
{

class GWPCA: public SpatialMonoscaleAlgorithm, public IMultivariableAnalysis
{
private:
    typedef arma::mat (GWPCA::*Solver)(const arma::mat&, arma::cube&, arma::mat&);

public: // Constructors and Deconstructors

    /**
     * @brief Construct a new GWPCA object.
     * 
     * Use gwmodel_create_gwpca_algorithm() to construct an instance in shared build.
     */
    GWPCA() {}

    /**
     * @brief Construct a new GWPCA object.
     * 
     * Use gwmodel_create_gwpca_algorithm() to construct an instance in shared build.
     */
    GWPCA(const arma::mat x, const arma::mat coords, const SpatialWeight& spatialWeight)
        : SpatialMonoscaleAlgorithm(spatialWeight, coords)
    {
        mX = x;
    }
    
    /**
     * @brief Destroy the GWPCA object.
     * 
     * Use gwmodel_delete_gwpca_algorithm() to destory an instance in shared build.
     */
    virtual ~GWPCA() {}

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
     * @return arma::mat Local Principle Values matrix.
     */
    arma::mat localPV() { return mLocalPV; }

    /**
     * @brief Get the Loadings matrix.
     * 
     * @return arma::mat Loadings matrix.
     */
    arma::cube loadings() { return mLoadings; }

    /**
     * @brief Get the Standard deviation matrix.
     * 
     * @return arma::mat Standard deviation matrix.
     */
    arma::mat sdev() { return mSDev; }

    /**
     * @brief Get the Scores matrix.
     * 
     * @return arma::mat Scores matrix.
     */
    arma::cube scores() { return mScores; }

public: // IMultivariableAnalysis
    virtual arma::mat variables() const override { return mX; }
    virtual void setVariables(const arma::mat& x) override { mX = x; }
    virtual void run() override;

public: // Algorithm
    virtual bool isValid() override;

private:

    /**
     * @brief Function to carry out PCA.
     * 
     * @param x Symmetric data matrix.
     * @param loadings Out reference to loadings matrix.
     * @param sdev Out reference to standard deviation matrix.
     * @return arma::mat Principle values matrix.
     */
    arma::mat pca(const arma::mat& x, arma::cube& loadings, arma::mat& sdev)
    {
        return (this->*mSolver)(x, loadings, sdev);
    }

    /**
     * @brief Serial version of PCA funtion.
     * 
     * @param x Symmetric data matrix.
     * @param loadings Out reference to loadings matrix.
     * @param sdev Out reference to standard deviation matrix.
     * @return arma::mat Principle values matrix.
     */
    arma::mat solveSerial(const arma::mat& x, arma::cube& loadings, arma::mat& sdev);

    /**
     * @brief Function to carry out weighted PCA.
     * 
     * @param x Symmetric data matrix.
     * @param w Weight vector.
     * @param V Right orthogonal matrix.
     * @param d Rectangular diagonal matrix
     */
    void wpca(const arma::mat& x, const arma::vec& w, arma::mat& V, arma::vec & d);

private:    // Algorithm Parameters
    int mK = 2;  //< Number of components to be kept.
    // bool mRobust = false;

private:    // Algorithm Results
    arma::mat mLocalPV;               //< Local principle component values.
    arma::cube mLoadings;             //< Loadings for each component.
    arma::mat mSDev;                  //< Standard Deviation.
    arma::cube mScores;               //< Scores for each variable.
    arma::uvec mWinner;               //< Winner variable at each sample.

private:    // Algorithm Runtime Variables
    arma::mat mX;
    arma::vec mLatestWt;

    Solver mSolver = &GWPCA::solveSerial;
};

}

#endif  // GWPCA_H