#ifndef GWMGWPCATASKTHREAD_H
#define GWMGWPCATASKTHREAD_H

#include <QObject>
#include "TaskThread/gwmspatialmonoscalealgorithm.h"
#include "TaskThread/imultivariableanalysis.h"
#include "TaskThread/iparallelable.h"

#include "TaskThread/gwmbandwidthsizeselector.h"
#include <gwmodel.h>

class GwmGWPCATaskThread : public GwmSpatialMonoscaleAlgorithm, public IBandwidthSizeSelectable, public IGwmMultivariableAnalysis, public IOpenmpParallelable, public gwm::IBandwidthSelectable
{
    Q_OBJECT
    enum BandwidthSelectionCriterionType
    {
        CV
    };

    typedef QList<QPair<QString, mat> > CreateResultLayerData;
    typedef QList<QPair<QString, vec> > CreatePlotLayerData;

    typedef double (GwmGWPCATaskThread::*BandwidthSelectCriterionFunction)(GwmBandwidthWeight*);

public:
    GwmGWPCATaskThread();

public:
    QList<GwmVariable> variables() const override 
    {
        return QList<GwmVariable>();
    };
    void setVariables(const QList<GwmVariable> &variables) override;
    void setVariables(const QList<GwmVariable> &&variables)
    {
        mVariables = variables;
    };

    bool isAutoselectBandwidth() const;
    void setIsAutoselectBandwidth(bool isAutoselectBandwidth);
    BandwidthSelectionCriterionType bandwidthSelectionCriterionType() const;
    void setBandwidthSelectionCriterionType(const BandwidthSelectionCriterionType &bandwidthSelectionCriterionType);

    void setOmpThreadNum(const int threadNum) override
    {
        mOmpThreadNum = threadNum;
    };

    BandwidthCriterionList bandwidthSelectorCriterions() const
    {
        return mSelector.bandwidthCriterion();
    }

    double k() const;
    void setK(double k);

    mat localPV() const;

    mat variance() const;
    
    mat sdev() const;

    cube loadings() const;

    cube scores() const;

public:     // QThread interface
    void run() override;


public:     // GwmTaskThread interface
    QString name() const override { return tr("GWPCA"); }


public:     // GwmSpatialMonoscaleAlgorithm interface
    bool isValid() override;


public:  // IParallelalbe interface
    int parallelAbility() const override;
    virtual ParallelType parallelType() const override;
    virtual void setParallelType(const ParallelType& type) override;

public:  // IOpenmpParallelable interface
    void setThreadNum(const int threadNum){};
    //void setOmpThreadNum(const int threadNum){};

    bool zscore() const;
    void setZscore(bool zscore);

    bool scoresCal() const;
    void setScoresCal(bool scoresCal);

    bool Robust() const;
    void setRobust(bool robust);

    bool getPlot() const;
    void setPlot(bool plot);

    void setCanceled(bool canceled) override;

private:
    void initPoints();
    void initXY(mat& x, const QList<GwmVariable>& indepVars);
    void variableZscore(mat& x);
    void calculateScores();

    void createResultLayer(CreateResultLayerData data,QList<QString> winvar);
    void createPlotLayer(CreatePlotLayerData data, QList<QString> varpc);
    std::unique_ptr<gwm::GWPCA> mAlgorithm;
    
    // wpca函数保留，用于带宽选择（内核库的wpca是私有的，无法直接访问）
    void wpca(const mat &x, const vec &wt, mat &V, vec &S);

    // Robust weighted PCA 函数
    void rwpca(const mat &x, const vec &wt, mat &V, vec &S);

    // Robust GWPCA 求解函数
    mat robustSolveSerial(const mat& x, cube& loadings, mat& sdev);

    double bandwidthSizeCriterionCVSerial(GwmBandwidthWeight* weight);
#ifdef ENABLE_OpenMP
    double bandwidthSizeCriterionCVOmp(GwmBandwidthWeight* weight);
#endif
    double criterion(GwmBandwidthWeight *weight) override
    {
        return (this->*mBandwidthSelectCriterionFunction)(weight);
    }

public:  // gwm::IBandwidthSelectable interface
    gwm::Status getCriterion(gwm::BandwidthWeight* weight, double& criterion) override;

private:
    QList<GwmVariable> mVariables;
    mat mDataPoints;

    int mK = 2;

    mat mX;
    vec mLatestWt;

    gwm::BandwidthSelector mSelector;
    BandwidthSelectionCriterionType mBandwidthSelectionCriterionType = BandwidthSelectionCriterionType::CV;
    BandwidthSelectCriterionFunction mBandwidthSelectCriterionFunction = &GwmGWPCATaskThread::bandwidthSizeCriterionCVSerial;
    bool mIsAutoselectBandwidth = false;

    // IOpenmpParallelable interface
    IParallelalbe::ParallelType mParallelType = IParallelalbe::ParallelType::SerialOnly;
    int mOmpThreadNum = 8;

    mat mLocalPV;
    mat mVariance;
    mat mSDev;
    cube mLoadings;
    cube mScores;
    cube mScoresFromKernel = cube();  // 从内核库获取的scores，用于对比

    bool mZscore;
    bool mScoresCal;


public:
    static int treeChildCount;

private:
    //用户选择RobustGWPCA
    bool mRobust;

    //用户选择GlyphPlot
    bool mPlot;

};

inline mat GwmGWPCATaskThread::variance() const
{
    return mVariance;
}

inline mat GwmGWPCATaskThread::sdev() const
{
    return mSDev;
}

inline int GwmGWPCATaskThread::parallelAbility() const
{
    return IParallelalbe::SerialOnly
        #ifdef ENABLE_OpenMP
            | IParallelalbe::OpenMP
        #endif
            ;
}

inline IParallelalbe::ParallelType GwmGWPCATaskThread::parallelType() const
{
    return mParallelType;
}

inline mat GwmGWPCATaskThread::localPV() const
{
    return mLocalPV;
}

inline double GwmGWPCATaskThread::k() const
{
    return mK;
}

inline cube GwmGWPCATaskThread::scores() const
{
    return mScores;
}

inline cube GwmGWPCATaskThread::loadings() const
{
    return mLoadings;
}

inline void GwmGWPCATaskThread::setK(double k)
{
    mK = k;
}

#endif // GWMGWPCATASKTHREAD_H
