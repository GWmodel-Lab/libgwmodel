#ifndef CGWMGWDR_H
#define CGWMGWDR_H

#include <vector>
#include <armadillo>
#include <gsl/gsl_vector.h>
#include "CGwmSpatialAlgorithm.h"
#include "spatialweight/CGwmSpatialWeight.h"
#include "IGwmRegressionAnalysis.h"
#include "CGwmVariableForwardSelector.h"
#include "IGwmParallelizable.h"

using namespace std;
using namespace arma;


class CGwmGWDR : public CGwmSpatialAlgorithm, public IGwmRegressionAnalysis, public IGwmVarialbeSelectable, public IGwmOpenmpParallelizable
{
public:
    typedef mat (CGwmGWDR::*RegressionCalculator)(const mat&, const vec&);

    typedef mat (CGwmGWDR::*RegressionHatmatrixCalculator)(const mat&, const vec&, mat&, vec&, vec&, mat&);


    enum NameFormat
    {
        Fixed,
        VarName,
        PrefixVarName,
        SuffixVariable
    };

    typedef tuple<string, mat, NameFormat> ResultLayerDataItem;

    enum BandwidthCriterionType
    {
        CV,
        AIC
    };

    typedef double (CGwmGWDR::*BandwidthCriterionCalculator)(const vector<CGwmBandwidthWeight*>&);

    typedef double (CGwmGWDR::*IndepVarCriterionCalculator)(const vector<GwmVariable>&);

public:
    static GwmRegressionDiagnostic CalcDiagnostic(const mat& x, const vec& y, const mat& betas, const vec& shat);

    static vec Fitted(const mat& x, const mat& betas)
    {
        return sum(betas % x, 1);
    }

    static double RSS(const mat& x, const mat& y, const mat& betas)
    {
        vec r = y - Fitted(x, betas);
        return sum(r % r);
    }

    static double AICc(const mat& x, const mat& y, const mat& betas, const vec& shat)
    {
        double ss = RSS(x, y, betas), n = (double)x.n_rows;
        return n * log(ss / n) + n * log(2 * datum::pi) + n * ((n + shat(0)) / (n - 2 - shat(0)));
    }

public:
    mat betas() const
    {
        return mBetas;
    }

    bool hasHatMatrix() const
    {
        return mHasHatMatrix;
    }

    void setHasHatMatrix(bool flag)
    {
        mHasHatMatrix = flag;
    }

    vector<CGwmSpatialWeight> spatialWeights()
    {
        return mSpatialWeights;
    }

    void setSpatialWeights(vector<CGwmSpatialWeight> spatialWeights)
    {
        mSpatialWeights = spatialWeights;
    }

    bool enableBandwidthOptimize()
    {
        return mEnableBandwidthOptimize;
    }

    void setEnableBandwidthOptimize(bool flag)
    {
        mEnableBandwidthOptimize = flag;
    }

    double bandwidthOptimizeEps() const
    {
        return mBandwidthOptimizeEps;
    }

    void setBandwidthOptimizeEps(double value)
    {
        mBandwidthOptimizeEps = value;
    }

    size_t bandwidthOptimizeMaxIter() const
    {
        return mBandwidthOptimizeMaxIter;
    }

    void setBandwidthOptimizeMaxIter(size_t value)
    {
        mBandwidthOptimizeMaxIter = value;
    }

    double bandwidthOptimizeStep() const
    {
        return mBandwidthOptimizeStep;
    }

    void setBandwidthOptimizeStep(double value)
    {
        mBandwidthOptimizeStep = value;
    }

    BandwidthCriterionType bandwidthCriterionType() const
    {
        return mBandwidthCriterionType;
    }

    void setBandwidthCriterionType(const BandwidthCriterionType& type);

    bool enableIndpenVarSelect() const
    {
        return mEnableIndepVarSelect;
    }

    void setEnableIndepVarSelect(bool flag)
    {
        mEnableIndepVarSelect = flag;
    }

    VariablesCriterionList indepVarCriterionList() const
    {
        return mIndepVarCriterionList;
    }

public: // CGwmAlgorithm
    void run();
    bool isValid()
    {
        // [TODO]: Add actual check codes.
        return true;
    }

public: // IGwmRegressionAnalysis
    GwmVariable dependentVariable() const
    {
        return mDepVar;
    }

    void setDependentVariable(const GwmVariable& variable)
    {
        mDepVar = variable;
    }

    vector<GwmVariable> independentVariables() const
    {
        return mIndepVars;
    }

    void setIndependentVariables(const vector<GwmVariable>& variables)
    {
        mIndepVars = variables;
    }

public:  // IRgressionAnalysis

    mat regression(const mat& x, const vec& y)
    {
        return (this->*mRegressionFunction)(x, y);
    }

    mat regressionHatmatrix(const mat& x, const vec& y, mat& betasSE, vec& shat, vec& qdiag, mat& S)
    {
        return (this->*mRegressionHatmatrixFunction)(x, y, betasSE, shat, qdiag, S);
    }

    GwmRegressionDiagnostic diagnostic() const
    {
        return mDiagnostic;
    }

public:  // IGwmVariableSelectable
    double getCriterion(const vector<GwmVariable>& variables) override
    {
        return (this->*mIndepVarCriterionFunction)(variables);
    }

public:  // IGwmOpenmpParallelizable
    int parallelAbility() const
    {
        return ParallelType::SerialOnly
        #ifdef ENABLE_OPENMP
        | ParallelType::OpenMP
        #endif        
        ;
    }
    
    ParallelType parallelType() const
    {
        return mParallelType;
    }
    
    void setParallelType(const ParallelType& type);

    void setOmpThreadNum(const int threadNum)
    {
        mOmpThreadNum = threadNum;
    }


public:
    double bandwidthCriterion(const vector<CGwmBandwidthWeight*>& bandwidths)
    {
        return (this->*mBandwidthCriterionFunction)(bandwidths);
    }

protected:
    mat regressionSerial(const mat& x, const vec& y);
    mat regressionOmp(const mat& x, const vec& y);
    mat regressionHatmatrixSerial(const mat& x, const vec& y, mat& betasSE, vec& shat, vec& qdiag, mat& S);
    mat regressionHatmatrixOmp(const mat& x, const vec& y, mat& betasSE, vec& shat, vec& qdiag, mat& S);

    double bandwidthCriterionCVSerial(const vector<CGwmBandwidthWeight*>& bandwidths);
    double bandwidthCriterionCVOmp(const vector<CGwmBandwidthWeight*>& bandwidths);
    double bandwidthCriterionAICSerial(const vector<CGwmBandwidthWeight*>& bandwidths);
    double bandwidthCriterionAICOmp(const vector<CGwmBandwidthWeight*>& bandwidths);

    double indepVarCriterionSerial(const vector<GwmVariable>& indepVars);
    double indepVarCriterionOmp(const vector<GwmVariable>& indepVars);

protected:
    void createResultLayer(initializer_list<ResultLayerDataItem> items);

private:
    void setXY(mat& x, mat& y, const CGwmSimpleLayer* layer, const GwmVariable& depVar, const vector<GwmVariable>& indepVars);

    bool isStoreS()
    {
        return mHasHatMatrix && (mSourceLayer->featureCount() < 8192);
    }

private:
    vector<CGwmSpatialWeight> mSpatialWeights;

    vec mY;
    mat mX;
    mat mBetas;
    GwmVariable mDepVar;
    vector<GwmVariable> mIndepVars;
    bool mHasHatMatrix;
    GwmRegressionDiagnostic mDiagnostic;

    RegressionCalculator mRegressionFunction = &CGwmGWDR::regressionSerial;
    RegressionHatmatrixCalculator mRegressionHatmatrixFunction = &CGwmGWDR::regressionHatmatrixSerial;

    bool mEnableBandwidthOptimize = false;
    BandwidthCriterionType mBandwidthCriterionType = BandwidthCriterionType::CV;
    BandwidthCriterionCalculator mBandwidthCriterionFunction = &CGwmGWDR::bandwidthCriterionCVSerial;
    double mBandwidthOptimizeEps = 1e-6;
    size_t mBandwidthOptimizeMaxIter = 100000;
    double mBandwidthOptimizeStep = 0.01;

    bool mEnableIndepVarSelect = false;
    double mIndepVarSelectThreshold = 3.0;
    VariablesCriterionList mIndepVarCriterionList;
    IndepVarCriterionCalculator mIndepVarCriterionFunction = &CGwmGWDR::indepVarCriterionSerial;

    ParallelType mParallelType = ParallelType::SerialOnly;
    int mOmpThreadNum = 8;
};


class CGwmGWDRBandwidthOptimizer
{
public:
    struct Parameter
    {
        CGwmGWDR* instance;
        vector<CGwmBandwidthWeight*>* bandwidths;
        uword featureCount;
    };

    static double criterion_function(const gsl_vector* bws, void* params);

public:
    CGwmGWDRBandwidthOptimizer(vector<CGwmBandwidthWeight*> weights)
    {
        mBandwidths = weights;
    }

    const int optimize(CGwmGWDR* instance, uword featureCount, size_t maxIter, double eps, double step);

private:
    vector<CGwmBandwidthWeight*> mBandwidths;
    Parameter mParameter;
};

#endif  // CGWMGWDR_H