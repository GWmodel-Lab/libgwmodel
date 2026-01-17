#include "gwmgwpcataskthread.h"
#include <SpatialWeight/gwmcrsdistance.h>
#include "TaskThread/gwmgeographicalweightedregressionalgorithm.h"
#include "gwmtaskthread.h"
#ifdef ENABLE_OpenMP
#include <omp.h>
#endif
#include <armadillo>
#include<math.h>
#include <qgsmaptool.h>
#include <QtDebug>
#include <QString>
#include <exception>
#include <stdexcept>

int GwmGWPCATaskThread::treeChildCount = 0;

GwmGWPCATaskThread::GwmGWPCATaskThread() : GwmSpatialMonoscaleAlgorithm(),
    mAlgorithm(std::make_unique<gwm::GWPCA>())
{

}

void GwmGWPCATaskThread::setCanceled(bool canceled)
{
    // mSelector.setCanceled(canceled);
    // mSpatialWeight.distance()->setCanceled(canceled);
    return GwmTaskThread::setCanceled(canceled);
}

void GwmGWPCATaskThread::run()
{
    if(!checkCanceled())
    {
        emit message(QString(tr("Setting data points ...")));
        initPoints();
        
        emit message(QString(tr("Setting X and Y.")));
        initXY(mX,mVariables);

        if(zscore()){
            emit message(QString(tr("Zscore normalizaiton...")));
            variableZscore(mX);
        }
    }   
    
    if(mIsAutoselectBandwidth && !checkCanceled())
    {
        emit message(QString(tr("Automatically selecting bandwidth ...")));
        emit tick(0, 0);
        
        if ((mSpatialWeight.distance()->type() == gwm::Distance::CRSDistance || 
             mSpatialWeight.distance()->type() == gwm::Distance::MinkwoskiDistance) && !checkCanceled())
        {
            gwm::CRSDistance* d = static_cast<gwm::CRSDistance*>(mSpatialWeight.distance());
            if(d)
            {
                d->makeParameter({ mDataPoints, mDataPoints });
            }
        }
        
        gwm::BandwidthWeight* bandwidthWeight0 = mSpatialWeight.weight<gwm::BandwidthWeight>();
        if(!bandwidthWeight0)
        {
            qDebug() << "[GWPCA] ERROR: Cannot get bandwidth weight!";
            emit error(tr("Cannot get bandwidth weight for bandwidth selection."));
            return;
        }
        
        double tmpMaxD = mSpatialWeight.distance()->maxDistance();
        double lower = bandwidthWeight0->adaptive() ? 2 : tmpMaxD / 5000;
        double upper = bandwidthWeight0->adaptive() ? mDataPoints.n_rows : tmpMaxD;
        
        if(bandwidthWeight0->bandwidth() <= 0 || 
           (bandwidthWeight0->adaptive() && bandwidthWeight0->bandwidth() < lower) ||
           (!bandwidthWeight0->adaptive() && bandwidthWeight0->bandwidth() < lower))
        {
            double initialBandwidth = bandwidthWeight0->adaptive() ? 
                std::max(2.0, std::min(20.0, (double)mDataPoints.n_rows * 0.1)) : 
                std::max(lower, tmpMaxD * 0.1);
            bandwidthWeight0->setBandwidth(initialBandwidth);
        }
        
        mSelector.setBandwidth(bandwidthWeight0);
        mSelector.setLower(lower);
        mSelector.setUpper(upper);
        
        try
        {
            gwm::BandwidthWeight* bandwidthWeight = mSelector.optimize(this);
            if(bandwidthWeight && !checkCanceled())
            {
                mSpatialWeight.setWeight(bandwidthWeight);
                mSelector.setBandwidth(bandwidthWeight);

                gwm::BandwidthWeight* verifyBw = mSpatialWeight.weight<gwm::BandwidthWeight>();
                if(verifyBw && verifyBw->bandwidth() == 0)
                {
                    qDebug() << "[GWPCA] ERROR: Bandwidth is 0 after setWeight! This may cause display issues.";
                }
            }
            else if(!bandwidthWeight)
            {
                qDebug() << "[GWPCA] WARNING: Bandwidth optimization returned NULL";
                emit error(tr("Bandwidth optimization failed: no optimal bandwidth found."));
                return;
            }
        }
        catch(const std::exception& e)
        {
            qDebug() << "[GWPCA] EXCEPTION during bandwidth optimization:" << e.what();
            emit error(QString(tr("Bandwidth optimization error: %1")).arg(e.what()));
            return;
        }
        catch(...)
        {
            qDebug() << "[GWPCA] UNKNOWN EXCEPTION during bandwidth optimization";
            emit error(tr("Unknown error occurred during bandwidth optimization."));
            return;
        }
    }
    
    if(!checkCanceled())
    {
        emit message(QString(tr("Principle components analyzing ...")));

        try
        {
            if(Robust())
            {
                emit message(QString(tr("Running Robust GWPCA ...")));
                mLocalPV = robustSolveSerial(mX, mLoadings, mSDev);
                
                if(checkCanceled())
                {
                    return;
                }
                
                mVariance = mSDev % mSDev;
            }
            else
            {
                if(!mAlgorithm)
                {
                    qDebug() << "[GWPCA] ERROR: mAlgorithm is NULL!";
                    emit error(tr("GWPCA algorithm object is not initialized."));
                    return;
                }
                
                mAlgorithm->setCoords(mDataPoints);
                mAlgorithm->setVariables(mX);
                mAlgorithm->setSpatialWeight(mSpatialWeight);
                mAlgorithm->setKeepComponents(mK);
                
                if(!mAlgorithm->isValid())
                {
                    qDebug() << "[GWPCA] ERROR: Algorithm configuration is invalid!";
                    emit error(tr("GWPCA algorithm configuration is invalid."));
                    return;
                }
                
                mAlgorithm->setTelegram(std::make_unique<GwmTaskThreadTelegram>(this));
                mAlgorithm->run();
                
                auto status = mAlgorithm->status();
                if(status != gwm::Status::Success)
                {
                    qDebug() << "[GWPCA] ERROR: Algorithm execution failed with status:" << static_cast<int>(status);
                    emit error(tr("GWPCA algorithm execution failed."));
                    return;
                }
                
                mLocalPV = mAlgorithm->localPV();
                mLoadings = mAlgorithm->loadings();
                mSDev = mAlgorithm->sdev();
                mVariance = mSDev % mSDev;
                
                // Get scores from kernel library for comparison
                if(scoresCal() && mDataPoints.n_rows <= 4096)
                {
                    const cube& kernelScores = mAlgorithm->scores();
                    mScoresFromKernel = kernelScores;
                    
                    qDebug() << "[GWPCA] Got scores from kernel library, dimensions:" << mScoresFromKernel.n_rows << "x" << mScoresFromKernel.n_cols << "x" << mScoresFromKernel.n_slices;
                }
            }

            if(scoresCal() && mDataPoints.n_rows <= 4096)
            {
                // Keep original calculateScores() result for comparison
                calculateScores();
                
                // Output comparison information
                if(!Robust() && mScoresFromKernel.n_elem > 0)
                {
                    qDebug() << "[GWPCA] ========== Scores Comparison Info ==========";
                    qDebug() << "[GWPCA] Original code scores dimensions:" << mScores.n_rows << "x" << mScores.n_cols << "x" << mScores.n_slices;
                    qDebug() << "[GWPCA] Kernel library scores dimensions:" << mScoresFromKernel.n_rows << "x" << mScoresFromKernel.n_cols << "x" << mScoresFromKernel.n_slices;
                    qDebug() << "[GWPCA] Note: Original code scores format is (nDp, mK, nDp), kernel library scores format is (nDp, nDp, mK)";
                    qDebug() << "[GWPCA] Original code: mScores.slice(i) is the score matrix (nDp, mK) for point i";
                    qDebug() << "[GWPCA] Kernel library: mScoresFromKernel.slice(j) is the score matrix (nDp, nDp) for component j";
                    
                    int sampleSize = std::min(5, (int)mDataPoints.n_rows);
                    int sampleComps = std::min(mK, 3);
                    qDebug() << "[GWPCA] First" << sampleSize << "points score comparison:";
                    for(int i = 0; i < sampleSize; i++)
                    {
                        qDebug() << "[GWPCA]   Point" << i << "scores:";
                        qDebug() << "[GWPCA]     Original code (mScores.slice(" << i << ")):";
                        for(int j = 0; j < sampleComps; j++)
                        {
                            vec scoreVec = mScores.slice(i).col(j);
                            QString scoreStr = "[";
                            for(uword k = 0; k < std::min((uword)5, scoreVec.n_elem); k++)
                            {
                                if(k > 0) scoreStr += ", ";
                                scoreStr += QString::number(scoreVec(k), 'f', 6);
                            }
                            if(scoreVec.n_elem > 5) scoreStr += ", ...";
                            scoreStr += "]";
                            qDebug() << "[GWPCA]       Component" << (j+1) << ":" << scoreStr;
                        }
                        qDebug() << "[GWPCA]     Kernel library (column" << i << "of each component):";
                        for(int j = 0; j < sampleComps; j++)
                        {
                            vec scoreVec = mScoresFromKernel.slice(j).col(i);
                            QString scoreStr = "[";
                            for(uword k = 0; k < std::min((uword)5, scoreVec.n_elem); k++)
                            {
                                if(k > 0) scoreStr += ", ";
                                scoreStr += QString::number(scoreVec(k), 'f', 6);
                            }
                            if(scoreVec.n_elem > 5) scoreStr += ", ...";
                            scoreStr += "]";
                            qDebug() << "[GWPCA]       Component" << (j+1) << ":" << scoreStr;
                        }
                    }

                    qDebug() << "[GWPCA] Format converted comparison:";
                    uword nDp = mDataPoints.n_rows;
                    cube kernelScoresConverted(nDp, mK, nDp, fill::zeros);
                    for(uword i = 0; i < nDp; i++)
                    {
                        for(int j = 0; j < mK; j++)
                        {
                            kernelScoresConverted.slice(i).col(j) = mScoresFromKernel.slice(j).col(i);
                        }
                    }
                    
                    for(int j = 0; j < sampleComps; j++)
                    {
                        double maxDiff = 0, meanDiff = 0;
                        uword diffCount = 0;
                        for(uword i = 0; i < nDp; i++)
                        {
                            vec diff = abs(mScores.slice(i).col(j) - kernelScoresConverted.slice(i).col(j));
                            double localMax = diff.max();
                            double localMean = mean(diff);
                            if(localMax > maxDiff) maxDiff = localMax;
                            meanDiff += localMean;
                            diffCount++;
                        }
                        meanDiff /= diffCount;
                        qDebug() << "[GWPCA]   Component" << (j+1) << "score difference - Max diff:" << maxDiff << "Mean diff:" << meanDiff;
                    }
                    
                    qDebug() << "[GWPCA] =====================================";
                }
            }
            else
            {
                mScoresCal = false;
            }
        }
        catch(const std::exception& e)
        {
            qDebug() << "[GWPCA] EXCEPTION caught:" << e.what();
            emit error(QString(tr("GWPCA execution error: %1")).arg(e.what()));
            return;
        }
        catch(...)
        {
            qDebug() << "[GWPCA] UNKNOWN EXCEPTION caught";
            emit error(tr("Unknown error occurred during GWPCA execution."));
            return;
        }
    }

    QList<QString> win_var_PC1;
    uvec iWinVar = index_max(mLoadings.slice(0), 1);
    for(int i = 0; i < mDataPoints.n_rows && !checkCanceled(); i++)
    {
        win_var_PC1.append(mVariables.at(iWinVar(i)).name);
    }
    
    if(!checkCanceled())
    {
        CreateResultLayerData resultLayerData = {
            qMakePair(QString("Comp.%1_PV"), mLocalPV),
            qMakePair(QString("local_CP"), sum(mLocalPV, 1))
        };
        if(scoresCal()){
            for(int i = 0; i < mK; i++){
                resultLayerData += qMakePair(QString("Scores%1"), mScores.slice(i));
            }
        };
        createResultLayer(resultLayerData,win_var_PC1);
        
        if(getPlot()){
            CreatePlotLayerData plotLayerData = {};
            vec vecLoadings(mDataPoints.n_rows * mVariables.size());
            QList<QString> var_pc;
            int v = 0;
            for(int i = 0; i < mDataPoints.n_rows; i++){
                for(int j=0; j < mVariables.size(); j++ ){
                    vecLoadings(v) = mLoadings.slice(0)(i,j);
                    var_pc.append(mVariables.at(j).name);
                    v++;
                }
            }
            plotLayerData += qMakePair(QStringLiteral("loadings"),vecLoadings);
            createPlotLayer(plotLayerData, var_pc);
        }
        emit success();
        emit tick(100, 100);
    }
    if(checkCanceled())
    {
        return;
    }
}

bool GwmGWPCATaskThread::isValid()
{
    gwm::BandwidthWeight* bandwidth = static_cast<gwm::BandwidthWeight*>(mSpatialWeight.weight());
    if(bandwidth){
        if(!mIsAutoselectBandwidth)
        {
            if(bandwidth->adaptive()){
                if(bandwidth->bandwidth() <= mVariables.size()){
                    return false;
                }
            }
        }
        if(mVariables.size() == 0){
            return false;
        }
        if(k()<=0 || k() > mVariables.size()){
            return false;
        }
    }else{
        return false;
    }
    return true;
}

void GwmGWPCATaskThread::initPoints()
{
    if(!mDataLayer)
    {
        qDebug() << "[GWPCA::initPoints] ERROR: Data layer is NULL!";
        return;
    }
    
    int nDp = mDataLayer->featureCount();
    mDataPoints = mat(nDp, 2, fill::zeros);
    
    QgsFeatureIterator iterator = mDataLayer->getFeatures();
    QgsFeature f;
    for (int i = 0; iterator.nextFeature(f); i++)
    {
        QgsPointXY centroPoint = f.geometry().centroid().asPoint();
        mDataPoints(i, 0) = centroPoint.x();
        mDataPoints(i, 1) = centroPoint.y();
    }
}

void GwmGWPCATaskThread::initXY(mat &x, const QList<GwmVariable> &indepVars)
{
    int nDp = mDataLayer->featureCount(), nVar = indepVars.size();
    
    x = mat(nDp, nVar, fill::zeros);
    
    QgsFeatureIterator iterator = mDataLayer->getFeatures();
    QgsFeature f;
    bool ok = false;
    int errorCount = 0;
    for (int i = 0; iterator.nextFeature(f); i++)
    {
        for (int k = 0; k < indepVars.size(); k++)
        {
            double vX = f.attribute(indepVars[k].name).toDouble(&ok);
            if (ok) 
            {
                x(i, k) = vX;
            }
            else 
            {
                errorCount++;
                if(errorCount <= 5)
                {
                    qDebug() << "[GWPCA::initXY] WARNING: Cannot convert variable" << indepVars[k].name << "at row" << i << "to number";
                }
                emit error(tr("Independent variable value cannot convert to a number. Set to 0."));
            }
        }
    }
}

void GwmGWPCATaskThread::variableZscore(mat& x)
{
    mat xmean = mean(x);
    mat xstd = stddev(x);
    for (int k = 0; k < x.n_cols; k++)
    {
        x.col(k) = (x.col(k) - xmean(k))/xstd(k);
    }
}


void GwmGWPCATaskThread::calculateScores()
{
    uword nDp = mDataPoints.n_rows, nVar = mVariables.size();
    mScores = cube(nDp, mK, nDp, fill::zeros);
    
    for(uword i = 0; i < nDp && !checkCanceled(); i++)
    {
        vec wt = mSpatialWeight.weightVector(i);
        uvec positive = find(wt > 0);
        vec newWt = wt.elem(positive);
        mat newX = mX.rows(positive);
        
        if(newWt.n_rows <= 5)
        {
            break;
        }
        
        mat V;
        vec d;
        if(!Robust()){
            wpca(newX, newWt, V, d);
        }else{
            rwpca(newX, newWt, V, d);
        }
        
        mat scorei(nDp, mK, fill::zeros);
        for(int j = 0; j < mK && !checkCanceled(); j++)
        {
            mat score = newX.each_row() % trans(V.col(j));
            scorei.col(j) = sum(score, 1);
        }
        mScores.slice(i) = scorei;
        emit tick(i + 1, nDp);
    }
}

void GwmGWPCATaskThread::wpca(const mat &x, const vec &wt, mat &V, vec &S)
{

    mat xw = x.each_col() % wt;
    mat centerized = (x.each_row() - sum(xw) / sum(wt)).each_col() % sqrt(wt);
    //SVD
    mat U;
    svd(U,S,V,centerized);
    //S即为R中的d
    //V即为R中的v
}

void GwmGWPCATaskThread::rwpca(const mat &x, const vec &wt, mat &V, vec &S)
{

    mat mids = x;
    mids = mids.each_row() - x.row((abs(wt - 0.5)).index_min());
    
    mat score;
    vec tsquared;
    princomp(V, score, S, tsquared, mids.each_col() % wt);

}

mat GwmGWPCATaskThread::robustSolveSerial(const mat& x, cube& loadings, mat& sdev)
{
    int nDp = mDataPoints.n_rows, nVar = mX.n_cols;

    mat d_all(nVar, nDp, fill::zeros);

    loadings = cube(nDp, nVar, mK, fill::zeros);
    
    for(int i=0;i<nDp && !checkCanceled();i++)
    {
        //vec distvi = mSpatialWeight.distance()->distance(i);
        vec wt = mSpatialWeight.weightVector(i);

        uvec positive = find(wt > 0);
        vec newWt = wt.elem(positive);
        mat newX = x.rows(positive);
        if(newWt.n_rows<=5)
        {
            break;
        }

        mat V;
        vec d;
        rwpca(newX,newWt,V,d);

        mLatestWt = newWt;
        d_all.col(i) = d;

        for(int j = 0; j < mK && !checkCanceled(); j++)
        {
            loadings.slice(j).row(i) = trans(V.col(j));
        }
        emit tick(i, nDp);
    }

    d_all = trans(d_all);
    mat variance = (d_all / pow(sum(mLatestWt),0.5)) % (d_all / pow(sum(mLatestWt),0.5));

    sdev = sqrt(variance);

    mat pv = variance.cols(0, mK - 1).each_col() % (1.0 / sum(variance,1)) * 100.0;
    return pv;
}

gwm::Status GwmGWPCATaskThread::getCriterion(gwm::BandwidthWeight* weight, double& criterion)
{
    if(checkCanceled())
    {
        criterion = DBL_MAX;
        return gwm::Status::Terminated;
    }
    
    // 将内核库的BandwidthWeight转换为本地的GwmBandwidthWeight
    GwmBandwidthWeight::KernelFunctionType kernelType = static_cast<GwmBandwidthWeight::KernelFunctionType>(weight->kernel());
    GwmBandwidthWeight localWeight(weight->bandwidth(), weight->adaptive(), kernelType);
    
    try
    {
        criterion = (this->*mBandwidthSelectCriterionFunction)(&localWeight);
        return gwm::Status::Success;
    }
    catch(const std::exception& e)
    {
        qDebug() << "[GWPCA::getCriterion] EXCEPTION:" << e.what();
        criterion = DBL_MAX;
        return gwm::Status::Success;
    }
    catch(...)
    {
        qDebug() << "[GWPCA::getCriterion] UNKNOWN EXCEPTION";
        criterion = DBL_MAX;
        return gwm::Status::Success;
    }
}

void GwmGWPCATaskThread::setBandwidthSelectionCriterionType(const GwmGWPCATaskThread::BandwidthSelectionCriterionType &bandwidthSelectionCriterionType)
{
    mBandwidthSelectionCriterionType = bandwidthSelectionCriterionType;
    QMap<QPair<BandwidthSelectionCriterionType, IParallelalbe::ParallelType>, BandwidthSelectCriterionFunction> mapper = {
        std::make_pair(qMakePair(BandwidthSelectionCriterionType::CV, IParallelalbe::ParallelType::SerialOnly), &GwmGWPCATaskThread::bandwidthSizeCriterionCVSerial),
    #ifdef ENABLE_OpenMP
        std::make_pair(qMakePair(BandwidthSelectionCriterionType::CV, IParallelalbe::ParallelType::OpenMP), &GwmGWPCATaskThread::bandwidthSizeCriterionCVOmp),
    #endif
        //std::make_pair(qMakePair(BandwidthSelectionCriterionType::CV, IParallelalbe::ParallelType::CUDA), &GwmGWPCATaskThread::bandwidthSizeCriterionCVCuda),
        //std::make_pair(qMakePair(BandwidthSelectionCriterionType::AIC, IParallelalbe::ParallelType::SerialOnly), &GwmGWPCATaskThread::bandwidthSizeCriterionAICSerial),
        //std::make_pair(qMakePair(BandwidthSelectionCriterionType::AIC, IParallelalbe::ParallelType::OpenMP), &GwmGWPCATaskThread::bandwidthSizeCriterionAICOmp),
        //std::make_pair(qMakePair(BandwidthSelectionCriterionType::AIC, IParallelalbe::ParallelType::CUDA), &GwmGWPCATaskThread::bandwidthSizeCriterionAICCuda)
    };
    mBandwidthSelectCriterionFunction = mapper[qMakePair(bandwidthSelectionCriterionType, mParallelType)];
}

bool GwmGWPCATaskThread::isAutoselectBandwidth() const
{
    return mIsAutoselectBandwidth;
}

void GwmGWPCATaskThread::setIsAutoselectBandwidth(bool isAutoselectBandwidth)
{
    mIsAutoselectBandwidth = isAutoselectBandwidth;
}

void GwmGWPCATaskThread::setVariables(const QList<GwmVariable> &variables)
{
    mVariables = variables;
}

void GwmGWPCATaskThread::setParallelType(const IParallelalbe::ParallelType &type)
{
    if(type & parallelAbility())
    {
        mParallelType = type;
        setBandwidthSelectionCriterionType(mBandwidthSelectionCriterionType);

    }
}

void GwmGWPCATaskThread::createResultLayer(CreateResultLayerData data, QList<QString> winvar)
{
    QgsVectorLayer* srcLayer = mDataLayer;
    QString layerFileName = QgsWkbTypes::displayString(srcLayer->wkbType()) + QStringLiteral("?");
    QString layerName = srcLayer->name();


    if(treeChildCount > 0)
    {
        if(!Robust()) layerName += QStringLiteral("_GWPCA") + "(" + QString::number(treeChildCount) + ")";
        else if(Robust()) layerName += QStringLiteral("_RGWPCA") + "(" + QString::number(treeChildCount) + ")";
    } else
    {
        if(!Robust()) layerName += QStringLiteral("_GWPCA");
        else if(Robust()) layerName += QStringLiteral("_RGWPCA");
    }

    treeChildCount++ ;


//    if(!Robust()){
//        layerName += QStringLiteral("_GWPCA");
//    }else{
//        layerName += QStringLiteral("_RGWPCA");
//    }

    mResultLayer = new QgsVectorLayer(layerFileName, layerName, QStringLiteral("memory"));
    mResultLayer->setCrs(srcLayer->crs());


    QgsFields fields;
    for (QPair<QString, const mat&> item : data)
    {
        QString title = item.first;
        const mat& value = item.second;
        if (value.n_cols > 1)
        {
            for (int k = 0; k < value.n_cols; k++)
            {
                QString fieldName = title.arg(k+1);
                fields.append(QgsField(fieldName, QVariant::Double, QStringLiteral("double")));
            }
        }
        else
        {
            fields.append(QgsField(title, QVariant::Double, QStringLiteral("double")));
        }
    }
    fields.append(QgsField("win_var_PC1",QVariant::String,QStringLiteral("varchar"),255));
    mResultLayer->dataProvider()->addAttributes(fields.toList());
    mResultLayer->updateFields();


    mResultLayer->startEditing();
    QgsFeatureIterator iterator = srcLayer->getFeatures();
    QgsFeature f;
    for (int i = 0; iterator.nextFeature(f); i++)
    {
        QgsFeature feature(fields);
        feature.setGeometry(f.geometry());

        int k = 0;
        for (QPair<QString, const mat&> item : data)
        {
            for (uword d = 0; d < item.second.n_cols; d++)
            {
                feature.setAttribute(k, item.second(i, d));
                k++;
            }
        }
        feature.setAttribute("win_var_PC1",winvar[i]);

        mResultLayer->addFeature(feature);
    }
    mResultLayer->commitChanges();
}

void GwmGWPCATaskThread::createPlotLayer(CreatePlotLayerData data, QList<QString> varpc)
{
    QgsVectorLayer* srcLayer = mDataLayer;
    QString layerFileName = QgsWkbTypes::displayString(srcLayer->wkbType()) + QStringLiteral("?");
    QString layerName = srcLayer->name();


    if(treeChildCount > 0)
    {
        layerName += QStringLiteral("_GlyphPlot") + "(" + QString::number(treeChildCount) + ")";
    } else
    {
        layerName += QStringLiteral("_GlyphPlot");
    }

    treeChildCount++ ;


    if(!Robust()){
        layerName += QStringLiteral("_GWPCA");
    }else{
        layerName += QStringLiteral("_RGWPCA");
    }

    mPlotLayer = new QgsVectorLayer(QgsWkbTypes::displayString(QgsWkbTypes::LineString), layerName, QStringLiteral("memory"));
    mPlotLayer->setCrs(srcLayer->crs());

    QgsFields fields;
//    for (QPair<QString, const mat&> item : data)
//    {
//        QString title = item.first;
//        const mat& value = item.second;
//        if (value.n_cols > 1)
//        {
//            for (int k = 0; k < value.n_cols; k++)
//            {
//                QString fieldName = title.arg(k+1);
//                fields.append(QgsField(fieldName, QVariant::Double, QStringLiteral("double")));
//            }
//        }
//        else
//        {
//            fields.append(QgsField(title, QVariant::Double, QStringLiteral("double")));
//        }
//    }
    fields.append(QgsField(QStringLiteral("loadings"), QVariant::Double, QStringLiteral("double")));
    fields.append(QgsField(QStringLiteral("var_name"),QVariant::String,QStringLiteral("varchar"),255));
    mPlotLayer->dataProvider()->addAttributes(fields.toList());
    mPlotLayer->updateFields();

    mPlotLayer->startEditing();
    QgsFeatureIterator iterator = srcLayer->getFeatures();
    QgsFeature f;
    double PI = 3.1415926535898;
    int k = 0;
    for (int i = 0; i<mVariables.size() * mDataPoints.n_rows; i++)
    {
//        QgsFeature feature(fields);
        QgsFeature lineFeature(fields);
//        std::cout<<i<<" x:"<<f.geometry().asPoint().x()<<" y:"<<f.geometry().asPoint().y()<<std::endl;
//        feature.setGeometry(f.geometry());
        if(i%4==0){
            k=0;
            iterator.nextFeature(f);
         }

        QgsPointXY sttPoint=f.geometry().asPoint();


        double angle = 2.0 * PI / mVariables.size();
//        for (QPair<QString, const mat&> item : data)
//        {
//            for (uword d = 0; d < item.second.n_cols; d++)
//            {
//                feature.setAttribute(k, item.second(i, d));
//                k++;
//            }
//        }

        for (QPair<QString, const vec&> item : data)
        {
            for (uword d = 0; d < item.second.n_cols; d++)
            {

                double x = sttPoint.x()+fabs(item.second(i, d)) * cos(k * angle)*10;
                double y = sttPoint.y()+fabs(item.second(i, d)) * sin(k * angle)*10;
//                std::cout<<fabs(item.second(i, d)) * cos(k * angle)<< "x: "<<x<<" y: "<<y<<std::endl;
                QVector<QgsPointXY> lineData ={};
                lineData+= sttPoint;
                lineData+= QgsPointXY(x,y);
//                std::cout<<"x: "<<x<<" y: "<<y<<std::endl;
                lineFeature.setGeometry(QgsGeometry::fromPolylineXY(lineData));
                lineFeature.setAttribute(0, item.second(i, d));
            }
        }

        k++;
        lineFeature.setAttribute("var_name",varpc[i]);

//        mPlotLayer->addFeature(feature);
        mPlotLayer->addFeature(lineFeature);
    }
    mPlotLayer->commitChanges();
    qWarning("test %s", (QgsWkbTypes::displayString(mPlotLayer->wkbType())).toStdString().data());
}

double GwmGWPCATaskThread::bandwidthSizeCriterionCVSerial(GwmBandwidthWeight *weight)
{
    int mBandwidthCounter = 0;
    int n = mX.n_rows;
    int m = mX.n_cols;
    double score = 0;

    if (mSpatialWeight.distance()->type() == gwm::Distance::CRSDistance || 
        mSpatialWeight.distance()->type() == gwm::Distance::MinkwoskiDistance)
    {
        gwm::CRSDistance* d = static_cast<gwm::CRSDistance*>(mSpatialWeight.distance());
        if(d)
        {
            d->makeParameter({ mDataPoints, mDataPoints });
        }
    }

    for (int i = 0; i < n && !checkCanceled(); i++)
    {
        vec distvi = mSpatialWeight.distance()->distance(i);
        if(distvi.n_elem == 0)
        {
            qDebug() << "[GWPCA::bandwidthSizeCriterionCVSerial] ERROR: Empty distance vector at point" << i;
            score = DBL_MAX;
            break;
        }
        
        vec wt = weight->weight(distvi);
        wt(i) = 0;

        uvec positive = find(wt > 0);
        if(positive.n_elem == 0)
        {
            qDebug() << "[GWPCA::bandwidthSizeCriterionCVSerial] WARNING: No positive weights at point" << i;
            score = DBL_MAX;
            break;
        }
        
        vec newWt = wt.elem(positive);
        mat newX = mX.rows(positive);
        //判断length(newWt)
        if(newWt.n_rows <= 1)
        {
            qDebug() << "[GWPCA::bandwidthSizeCriterionCVSerial] WARNING: Insufficient points at" << i << ", n_rows:" << newWt.n_rows;
            score = DBL_MAX;
            break;
        }
        
        mat V;
        vec S;
        try
        {
            wpca(newX, newWt, V, S);
            if(V.n_cols < mK)
            {
                qDebug() << "[GWPCA::bandwidthSizeCriterionCVSerial] WARNING: V.n_cols < mK at point" << i;
                score = DBL_MAX;
                break;
            }
            V = V.cols(0, mK - 1);
            V = V * trans(V);
            score = score + pow(sum(mX.row(i) - mX.row(i) * V),2);
        }
        catch(const std::exception& e)
        {
            qDebug() << "[GWPCA::bandwidthSizeCriterionCVSerial] EXCEPTION at point" << i << ":" << e.what();
            score = DBL_MAX;
            break;
        }
        
        mBandwidthCounter++;
        if (mBandwidthCounter < 10)
            emit tick(mBandwidthCounter * 10 + i * 5 / n, 100);
    }
    return score;
}
#ifdef ENABLE_OpenMP
double GwmGWPCATaskThread::bandwidthSizeCriterionCVOmp(GwmBandwidthWeight *weight)
{
    int n = mX.n_rows;
    int m = mX.n_cols;
    double score = 0;
    bool flag = true;
    vec score_all(mOmpThreadNum, fill::zeros);
    int current = 0;
#pragma omp parallel for num_threads(mOmpThreadNum)
    for (int i = 0; i < n; i++)
    {
        if(flag && !checkCanceled())
        {
            int thread = omp_get_thread_num();
            vec distvi = mSpatialWeight.distance()->distance(i);
            vec wt = weight->weight(distvi);
            wt(i) = 0;
            uvec positive = find(wt > 0);
            vec newWt = wt.elem(positive);
            mat newX = mX.rows(positive);
            //判断length(newWt)
            if(newWt.n_rows <=1)
            {
                flag=false;
            }else{
                mat V;
                vec S;
                // 带宽选择时总是使用普通GWPCA（wpca），无论是否Robust模式
                // 这样Robust GWPCA会使用与普通GWPCA相同的带宽
                wpca(newX, newWt, V, S);
                V = V.cols(0, mK - 1);
                V = V * trans(V);
                score_all(thread) += pow(sum(mX.row(i) - mX.row(i) * V),2);
            }
            if(mSelector.counter<10)
                emit tick(mSelector.counter * 10 + current * 10 / n, 100);
            current++;
        }
    }
    score = sum(score_all);
    return score;
}
#endif
bool GwmGWPCATaskThread::zscore() const
{
    return mZscore;
}

void GwmGWPCATaskThread::setZscore(bool zscore)
{
    mZscore = zscore;
}

bool GwmGWPCATaskThread::scoresCal() const
{
    return mScoresCal;
}

void GwmGWPCATaskThread::setScoresCal(bool scoresCal)
{
    mScoresCal = scoresCal;
}

bool GwmGWPCATaskThread::Robust() const
{
    return mRobust;
}

void GwmGWPCATaskThread::setRobust(bool robust)
{
    mRobust=robust;
}

bool GwmGWPCATaskThread::getPlot() const
{
    return mPlot;
}

void GwmGWPCATaskThread::setPlot(bool plot)
{
    mPlot=plot;
}

