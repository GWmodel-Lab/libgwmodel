#include "gwmmultiscalegwralgorithm.h"
#ifdef ENABLE_OpenMP
#include <omp.h>
#endif
#include <exception>
#include "GWmodel/GWmodel.h"
#include <SpatialWeight/gwmcrsdistance.h>

using namespace std;

int GwmMultiscaleGWRAlgorithm::treeChildCount = 0;

GwmEnumValueNameMapper<GwmMultiscaleGWRAlgorithm::BandwidthInitilizeType> GwmMultiscaleGWRAlgorithm::BandwidthInitilizeTypeNameMapper = {
    make_pair(GwmMultiscaleGWRAlgorithm::BandwidthInitilizeType::Null, tr("Not initilized, not specified")),
    make_pair(GwmMultiscaleGWRAlgorithm::BandwidthInitilizeType::Initial, tr("Initilized")),
    make_pair(GwmMultiscaleGWRAlgorithm::BandwidthInitilizeType::Specified, tr("Specified"))
};

GwmEnumValueNameMapper<GwmMultiscaleGWRAlgorithm::BandwidthSelectionCriterionType> GwmMultiscaleGWRAlgorithm::BandwidthSelectionCriterionTypeNameMapper = {
    make_pair(GwmMultiscaleGWRAlgorithm::BandwidthSelectionCriterionType::CV, tr("CV")),
    make_pair(GwmMultiscaleGWRAlgorithm::BandwidthSelectionCriterionType::AIC, tr("AIC"))
};

GwmEnumValueNameMapper<GwmMultiscaleGWRAlgorithm::BackFittingCriterionType> GwmMultiscaleGWRAlgorithm::BackFittingCriterionTypeNameMapper = {
    make_pair(GwmMultiscaleGWRAlgorithm::BackFittingCriterionType::CVR, tr("CVR")),
    make_pair(GwmMultiscaleGWRAlgorithm::BackFittingCriterionType::dCVR, tr("dCVR"))
};

GwmDiagnostic GwmMultiscaleGWRAlgorithm::CalcDiagnostic(const mat &x, const vec &y, const mat &S0, double RSS)
{
    // 诊断信息
    double nDp = x.n_rows;
    double RSSg = RSS;
    double sigmaHat21 = RSSg / nDp;
    double TSS = sum((y - mean(y)) % (y - mean(y)));
    double Rsquare = 1 - RSSg / TSS;

    double trS = trace(S0);
    double trStS = trace(S0.t() * S0);
    double enp = 2 * trS - trStS;
    double edf = nDp - 2 * trS + trStS;
    double AICc = nDp * log(sigmaHat21) + nDp * log(2 * M_PI) + nDp * ((nDp + trS) / (nDp - 2 - trS));
    double adjustRsquare = 1 - (1 - Rsquare) * (nDp - 1) / (edf - 1);

    // 保存结果
    GwmDiagnostic diagnostic;
    diagnostic.RSS = RSSg;
    diagnostic.AICc = AICc;
    diagnostic.ENP = enp;
    diagnostic.EDF = edf;
    diagnostic.RSquareAdjust = adjustRsquare;
    diagnostic.RSquare = Rsquare;
    return diagnostic;
}

GwmMultiscaleGWRAlgorithm::GwmMultiscaleGWRAlgorithm()
    : GwmSpatialMultiscaleAlgorithm()
{
}

void GwmMultiscaleGWRAlgorithm::setCanceled(bool canceled)
{
    selector.setCanceled(canceled);
    return GwmTaskThread::setCanceled(canceled);
}

//OLS计算代码
GwmBasicGWRAlgorithm::OLSVar GwmMultiscaleGWRAlgorithm::CalOLS(const mat &x, const vec &y){
    QMap<QString,QList<int> > Coefficients;
    double nVar = mX.n_cols;
    double np = x.n_rows;
    mat xt = x.t();
    mat betahat = inv(xt * x)*xt*y;
    vec yhat = x*betahat;
    double ymean = mean(y);
    double sst = sum((y-ymean).t()*(y-ymean));
    double ssr = sum((yhat-ymean).t()*(yhat-ymean));
    double sse = sum((y-yhat).t()*(y-yhat));
    double Rsquared = 1- sse/sst;
    double adjRsquared = 1-(sse/(np-1-nVar))/(sst/(np-1));
//    double Ft = (ssr/3)/(sse/100-2-1);
    vec rs = y-yhat;
    double rmean = mean(rs);
    double rsd = sqrt(abs((sum((rs-rmean).t()*(rs-rmean)))/np));
    mat c = inv((xt * x));
    vec cdiag = diagvec(c);
    double unb = sqrt((sse/np-1-nVar));
    double varRes = abs((sum((rs-rmean).t()*(rs-rmean)))/np);
    double ll = -(np/2)*log(2*datum::pi)-(np/2)*log(varRes)-np/2;
    double AIC = -2*ll + 2*(nVar+1);
    double AICC = AIC+2*nVar*(nVar+1)/(np-nVar-1);
    //结果赋予结构体
    QMap<QString,QList<double> > coeffs;
    for(int i = 0 ; i < nVar ; i++){
        QString variableName = i == 0 ? QStringLiteral("Intercept") : mIndepVars[i - 1].name;
        QList<double> coeff;
        coeff.append(betahat[i]);
        double std = unb*sqrt(cdiag[i]);
        coeff.append(std);
        double tvalue = betahat[i]/std;
        coeff.append(tvalue);
        coeffs[variableName]=coeff;
    }
    return {rsd,Rsquared,adjRsquared,coeffs,AIC,AICC};
}

void GwmMultiscaleGWRAlgorithm::run()
{
    if(!checkCanceled())
    {
        initPoints();
        initXY(mX, mY, mDepVar, mIndepVars);
    }
    uword nDp = mX.n_rows, nVar = mX.n_cols;

    // ********************************
    // Centering and scaling predictors
    // ********************************
    mX0 = mX;
    mY0 = mY;
    for (uword i = 1; i < nVar & !checkCanceled(); i++)
    {
        if (mPreditorCentered[i])
        {
            mX.col(i) = mX.col(i) - mean(mX.col(i));
        }
    }

    // ***********************
    // Intialize the bandwidth
    // ***********************
    mYi = mY;
    for (uword i = 0; i < nVar & !checkCanceled(); i++)
    {
        if (mBandwidthInitilize[i] == BandwidthInitilizeType::Null)
        {
            emit message(tr("Calculating the initial bandwidth for %1 ...").arg(i == 0 ? "Intercept" : mIndepVars[i-1].name));
            mBandwidthSizeCriterion = bandwidthSizeCriterionVar(mBandwidthSelectionApproach[i]);
            mBandwidthSelectionCurrentIndex = i;
            mXi = mX.col(i);
            GwmBandwidthWeight* bw0 = bandwidth(i);
            bool adaptive = bw0->adaptive();
            selector.setBandwidth(bw0);
            selector.setLower(adaptive ? mAdaptiveLower : 0.0);
            selector.setUpper(adaptive ? mDataLayer->featureCount() : mSpatialWeights[i].distance()->maxDistance());
            GwmBandwidthWeight* bw = selector.optimize(this);
            if (bw)
            {
                mSpatialWeights[i].setWeight(bw);
            }
        }
    }

    if(mOLS&&!checkCanceled()){
        mOLSVar = CalOLS(mX,mY);
    }

    // *****************************************************
    // Calculate the initial beta0 from the above bandwidths
    // *****************************************************
    emit message(tr("Calculating the initial beta0 from the above bandwidths ..."));
    GwmBandwidthWeight* bw0 = bandwidth(0);
    bool adaptive = bw0->adaptive();
    mBandwidthSizeCriterion = bandwidthSizeCriterionAll(mBandwidthSelectionApproach[0]);
    GwmBandwidthSizeSelector initBwSelector;
    initBwSelector.setBandwidth(bw0);
    initBwSelector.setLower(adaptive ? mAdaptiveLower : 0.0);
    initBwSelector.setUpper(adaptive ? mDataLayer->featureCount() : mSpatialWeights[0].distance()->maxDistance());
    GwmBandwidthWeight* initBw = initBwSelector.optimize(this);
    if (!initBw)
    {
        emit error(tr("Cannot select initial bandwidth."));
        return;
    }
    mInitSpatialWeight.setWeight(initBw);

    // 初始化诊断信息矩阵
    if (mHasHatMatrix && !checkCanceled())
    {
        mS0 = mat(nDp, nDp, fill::zeros);
        mSArray = cube(nDp, nDp, nVar, fill::zeros);
        mC = cube(nVar, nDp, nDp, fill::zeros);
    }

    mBetas = regression(mX, mY);

    if (mHasHatMatrix && !checkCanceled())
    {
        mDiagnostic = CalcDiagnostic(mX, mY, mS0, mRSS0);
        vec yhat = fitted(mX, mBetas);
        vec residual = mY - yhat;
        mBetasTV = mBetas / mBetasSE;
        createResultLayer({
            qMakePair(QString("%1"), mBetas),
            qMakePair(QString("yhat"), yhat),
            qMakePair(QString("residual"), residual),
            qMakePair(QString("%1_SE"),mBetasSE),
            qMakePair(QString("%1_TV"),mBetasTV)
        });
    }
    else
    {
        createResultLayer({
            qMakePair(QString("%1"), mBetas)
        });
    }

    if(!checkCanceled())
    {
        emit success();
        emit tick(100,100);
    }
    else return;
}

mat GwmMultiscaleGWRAlgorithm::regression(const mat &x, const vec &y)
{
    uword nDp = x.n_rows, nVar = x.n_cols;
    mat betas = (this->*mRegressionAll)(x, y);

    if (mHasHatMatrix && !checkCanceled())
    {
        mat idm(nVar, nVar, fill::eye);
        for (uword i = 0; i < nVar & !checkCanceled(); ++i)
        {
            for (uword j = 0; j < nDp & !checkCanceled(); ++j)
            {
                mSArray.slice(i).row(j) = x(j, i) * (idm.row(i) * mC.slice(j));
            }
        }
    }

    // ***********************************************************
    // Select the optimum bandwidths for each independent variable
    // ***********************************************************
    emit message(QString("-------- Select the Optimum Bandwidths for each Independent Varialbe --------"));
    uvec bwChangeNo(nVar, fill::zeros);
    vec resid = y - fitted(x, betas);
    double RSS0 = sum(resid % resid), RSS1 = DBL_MAX;
    double criterion = DBL_MAX;
    for (int iteration = 1; iteration <= mMaxIteration && criterion > mCriterionThreshold & !checkCanceled(); iteration++)
    {
        emit tick(iteration - 1, mMaxIteration);
        for (uword i = 0; i < nVar & !checkCanceled(); i++)
        {
            QString varName = i == 0 ? QStringLiteral("Intercept") : mIndepVars[i-1].name;
            vec fi = betas.col(i) % x.col(i);
            vec yi = resid + fi;
            if (mBandwidthInitilize[i] != BandwidthInitilizeType::Specified)
            {
                emit message(QString("Now select an optimum bandwidth for the variable: %1").arg(varName));
                mBandwidthSizeCriterion = bandwidthSizeCriterionVar(mBandwidthSelectionApproach[i]);
                mBandwidthSelectionCurrentIndex = i;
                mYi = yi;
                mXi = mX.col(i);
                GwmBandwidthWeight* bwi0 = bandwidth(i);
                bool adaptive = bwi0->adaptive();
                GwmBandwidthSizeSelector selector;
                selector.setBandwidth(bwi0);
                selector.setLower(adaptive ? mAdaptiveLower : 0.0);
                selector.setUpper(adaptive ? mDataLayer->featureCount() : mSpatialWeights[i].distance()->maxDistance());
                GwmBandwidthWeight* bwi = selector.optimize(this);
                double bwi0s = bwi0->bandwidth(), bwi1s = bwi->bandwidth();
                emit message(QString("The newly selected bandwidth for variable %1 is %2 (last is %3, difference is %4)")
                             .arg(varName).arg(bwi1s).arg(bwi0s).arg(abs(bwi1s - bwi0s)));
                if (abs(bwi1s - bwi0s) > mBandwidthSelectThreshold[i])
                {
                    bwChangeNo(i) = 0;
                    emit message(QString("The bandwidth for variable %1 will be continually selected in the next iteration").arg(varName));
                }
                else
                {
                    bwChangeNo(i) += 1;
                    if (bwChangeNo(i) >= mBandwidthSelectRetryTimes)
                    {
                        mBandwidthInitilize[i] = BandwidthInitilizeType::Specified;
                        emit message(QString("The bandwidth for variable %1 seems to be converged and will be kept the same in the following iterations.").arg(varName));
                    }
                    else
                    {
                        emit message(QString("The bandwidth for variable %1 seems to be converged for %2 times. It will be continually optimized in the next %3 times.")
                                     .arg(varName).arg(bwChangeNo(i)).arg(mBandwidthSelectRetryTimes - bwChangeNo(i)));
                    }
                }
                mSpatialWeights[i].setWeight(bwi);
            }

            mat S;
            betas.col(i) = (this->*mRegressionVar)(x.col(i), yi, i, S);
            if (mHasHatMatrix && !checkCanceled())
            {
                mat SArrayi = mSArray.slice(i);
                mSArray.slice(i) = S * SArrayi + S - S * mS0;
                mS0 = mS0 - SArrayi + mSArray.slice(i);
            }
            resid = y - Fitted(x, betas);
        }
        RSS1 = RSS(x, y, betas);
        criterion = (mCriterionType == BackFittingCriterionType::CVR) ?
                    abs(RSS1 - RSS0) :
                    sqrt(abs(RSS1 - RSS0) / RSS1);
        QString criterionName = mCriterionType == BackFittingCriterionType::CVR ? "change value of RSS (CVR)" : "differential change value of RSS (dCVR)";
        emit message(QString("Iteration %1 the %2 is %3").arg(iteration).arg(criterionName).arg(criterion));
        RSS0 = RSS1;
        emit message(QString("---- End of Iteration %1 ----").arg(iteration));
    }
    emit message(QString("-------- [End] Select the Optimum Bandwidths for each Independent Varialbe --------"));
    mRSS0 = RSS0;
    return betas;
}

bool GwmMultiscaleGWRAlgorithm::isValid()
{
    if (mIndepVars.size() < 1)
        return false;

    int nVar = mIndepVars.size() + 1;

    if (mSpatialWeights.size() != nVar)
        return false;

    if (mBandwidthInitilize.size() != nVar)
        return false;

    if (mBandwidthSelectionApproach.size() != nVar)
        return false;

    if (mPreditorCentered.size() != nVar)
        return false;

    if (mBandwidthSelectThreshold.size() != nVar)
        return false;

    for (int i = 0; i < nVar; i++)
    {
        GwmBandwidthWeight* bw = mSpatialWeights[i].weight<GwmBandwidthWeight>();
        if (mBandwidthInitilize[i] == GwmMultiscaleGWRAlgorithm::Specified || mBandwidthInitilize[i] == GwmMultiscaleGWRAlgorithm::Initial)
        {
            if (bw->adaptive())
            {
                if (bw->bandwidth() <= 1)
                    return false;
            }
            else
            {
                if (bw->bandwidth() < 0.0)
                    return false;
            }
        }
    }

    return true;
}

void GwmMultiscaleGWRAlgorithm::initPoints()
{
    int nDp = mDataLayer->featureCount();
    mDataPoints = mat(nDp, 2, fill::zeros);
    QgsFeatureIterator iterator = mDataLayer->getFeatures();
    QgsFeature f;
    for (int i = 0; i < nDp && iterator.nextFeature(f); i++)
    {
        QgsPointXY centroPoint = f.geometry().centroid().asPoint();
        mDataPoints(i, 0) = centroPoint.x();
        mDataPoints(i, 1) = centroPoint.y();
    }
    // Regression Layer
    if (hasRegressionLayer())
    {
        int nRp = mRegressionLayer->featureCount();
        mRegressionPoints = mat(nRp, 2, fill::zeros);
        QgsFeatureIterator iterator = mRegressionLayer->getFeatures();
        QgsFeature f;
        for (int i = 0; i < nRp && iterator.nextFeature(f); i++)
        {
            QgsPointXY centroPoint = f.geometry().centroid().asPoint();
            mRegressionPoints(i, 0) = centroPoint.x();
            mRegressionPoints(i, 1) = centroPoint.y();
        }
    }
    else mRegressionPoints = mDataPoints;
    // 设置空间距离中的数据指针
    if (mInitSpatialWeight.distance()->type() == GwmDistance::CRSDistance || mInitSpatialWeight.distance()->type() == GwmDistance::MinkwoskiDistance)
    {
        GwmCRSDistance* d = mInitSpatialWeight.distance<GwmCRSDistance>();
        d->setDataPoints(&mDataPoints);
        d->setFocusPoints(&mRegressionPoints);
    }
    for (const GwmSpatialWeight& sw : mSpatialWeights)
    {
        if (sw.distance()->type() == GwmDistance::CRSDistance || sw.distance()->type() == GwmDistance::MinkwoskiDistance)
        {
            GwmCRSDistance* d = sw.distance<GwmCRSDistance>();
            d->setDataPoints(&mDataPoints);
            d->setFocusPoints(&mRegressionPoints);
        }
    }
}

void GwmMultiscaleGWRAlgorithm::initXY(mat &x, mat &y, const GwmVariable &depVar, const QList<GwmVariable> &indepVars)
{
    int nDp = mDataLayer->featureCount(), nVar = indepVars.size() + 1;
    // Data layer and X,Y
    x = mat(nDp, nVar, fill::zeros);
    y = vec(nDp, fill::zeros);
    QgsFeatureIterator iterator = mDataLayer->getFeatures();
    QgsFeature f;
    bool ok = false;
    for (int i = 0; iterator.nextFeature(f); i++)
    {

        double vY = f.attribute(depVar.name).toDouble(&ok);
        if (ok)
        {
            y(i) = vY;
            x(i, 0) = 1.0;
            for (int k = 0; k < indepVars.size(); k++)
            {
                double vX = f.attribute(indepVars[k].name).toDouble(&ok);
                if (ok) x(i, k + 1) = vX;
                else emit error(tr("Independent variable value cannot convert to a number. Set to 0."));
            }
        }
        else emit error(tr("Dependent variable value cannot convert to a number. Set to 0."));
    }
}

void GwmMultiscaleGWRAlgorithm::createResultLayer(initializer_list<CreateResultLayerDataItem> data)
{
    emit message("Creating result layer...");
    QgsVectorLayer* srcLayer = mRegressionLayer ? mRegressionLayer : mDataLayer;
    QString layerFileName = QgsWkbTypes::displayString(srcLayer->wkbType()) + QStringLiteral("?");
    QString layerName = srcLayer->name();
    //避免图层名重复
    if(treeChildCount > 0)
    {
        layerName += QStringLiteral("_MGWR") + "(" + QString::number(treeChildCount) + ")";
    } else
    {
        layerName += QStringLiteral("_MGWR");
    }
    //节点记录标签
    treeChildCount++ ;


    mResultLayer = new QgsVectorLayer(layerFileName, layerName, QStringLiteral("memory"));
    mResultLayer->setCrs(srcLayer->crs());

    // 设置字段
    QgsFields fields;
    for (QPair<QString, const mat&> item : data)
    {
        QString title = item.first;
        const mat& value = item.second;
        if (value.n_cols > 1)
        {
            for (uword k = 0; k < value.n_cols; k++)
            {
                QString variableName = k == 0 ? QStringLiteral("Intercept") : mIndepVars[k - 1].name;
                QString fieldName = title.arg(variableName);
                fields.append(QgsField(fieldName, QVariant::Double, QStringLiteral("double")));
            }
        }
        else
        {
            fields.append(QgsField(title, QVariant::Double, QStringLiteral("double")));
        }
    }
    mResultLayer->dataProvider()->addAttributes(fields.toList());
    mResultLayer->updateFields();

    // 设置要素几何
    mResultLayer->startEditing();
    QgsFeatureIterator iterator = srcLayer->getFeatures();
    QgsFeature f;
    for (int i = 0; iterator.nextFeature(f); i++)
    {
        QgsFeature feature(fields);
        feature.setGeometry(f.geometry());

        // 设置属性
        int k = 0;
        for (QPair<QString, const mat&> item : data)
        {
            for (uword d = 0; d < item.second.n_cols; d++)
            {
                feature.setAttribute(k, item.second(i, d));
                k++;
            }
        }

        mResultLayer->addFeature(feature);
    }
    mResultLayer->commitChanges();
}


mat GwmMultiscaleGWRAlgorithm::regressionAllSerial(const mat& x, const vec& y)
{
    int nDp = x.n_rows, nVar = x.n_cols;
    mat betas(nVar, nDp, fill::zeros);
    if (mHasHatMatrix && !checkCanceled())
    {
        mat betasSE(nVar, nDp, fill::zeros);
        for (int i = 0; i < nDp & !checkCanceled(); i++)
        {
            vec w = mInitSpatialWeight.weightVector(i);
            mat xtw = trans(x.each_col() % w);
            mat xtwx = xtw * x;
            mat xtwy = xtw * y;
            try
            {
                mat xtwx_inv = inv_sympd(xtwx);
                betas.col(i) = xtwx_inv * xtwy;
                mat ci = xtwx_inv * xtw;
                betasSE.col(i) = sum(ci % ci, 1);
                mat si = x.row(i) * ci;
                mS0.row(i) = si;
                mC.slice(i) = ci;
            } catch (exception e) {
                emit error(e.what());
            }
        }
        mBetasSE = betasSE.t();
    }
    else
    {
        for (int i = 0; i < nDp & !checkCanceled(); i++)
        {
            vec w = mInitSpatialWeight.weightVector(i);
            mat xtw = trans(x.each_col() % w);
            mat xtwx = xtw * x;
            mat xtwy = xtw * y;
            try
            {
                mat xtwx_inv = inv_sympd(xtwx);
                betas.col(i) = xtwx_inv * xtwy;
            } catch (exception e) {
                emit error(e.what());
            }
        }
    }
    return betas.t();
}
#ifdef ENABLE_OpenMP
mat GwmMultiscaleGWRAlgorithm::regressionAllOmp(const mat &x, const vec &y)
{
    int nDp = x.n_rows, nVar = x.n_cols;
    mat betas(nVar, nDp, fill::zeros);
    if (mHasHatMatrix && !checkCanceled())
    {
        mat betasSE(nVar, nDp, fill::zeros);
#pragma omp parallel for num_threads(mOmpThreadNum)
        for (int i = 0; i < nDp; i++)
        {
            if(!checkCanceled())
            {
                vec w = mInitSpatialWeight.weightVector(i);
                mat xtw = trans(x.each_col() % w);
                mat xtwx = xtw * x;
                mat xtwy = xtw * y;
                try
                {
                    mat xtwx_inv = inv_sympd(xtwx);
                    betas.col(i) = xtwx_inv * xtwy;
                    mat ci = xtwx_inv * xtw;
                    betasSE.col(i) = sum(ci % ci, 1);
                    mat si = x.row(i) * ci;
                    mS0.row(i) = si;
                    mC.slice(i) = ci;
                } catch (exception e) {
                    emit error(e.what());
                }
            }
        }
        mBetasSE = betasSE.t();
    }
    else
    {
#pragma omp parallel for num_threads(mOmpThreadNum)
        for (int i = 0; i < nDp; i++)
        {
            if(!checkCanceled())
            {
                vec w = mInitSpatialWeight.weightVector(i);
                mat xtw = trans(x.each_col() % w);
                mat xtwx = xtw * x;
                mat xtwy = xtw * y;
                try
                {
                    mat xtwx_inv = inv_sympd(xtwx);
                    betas.col(i) = xtwx_inv * xtwy;
                } catch (exception e) {
                    emit error(e.what());
                }
            }
        }
    }
    return betas.t();
}
#endif
vec GwmMultiscaleGWRAlgorithm::regressionVarSerial(const vec &x, const vec &y, const int var, mat &S)
{
    int nDp = x.n_rows;
    mat betas(1, nDp, fill::zeros);
    if (mHasHatMatrix && !checkCanceled())
    {
        mat ci, si;
        S = mat(mHasHatMatrix ? nDp : 1, nDp, fill::zeros);
        for (int i = 0; i < nDp & !checkCanceled(); i++)
        {
            vec w = mSpatialWeights[var].weightVector(i);
            mat xtw = trans(x % w);
            mat xtwx = xtw * x;
            mat xtwy = xtw * y;
            try
            {
                mat xtwx_inv = inv_sympd(xtwx);
                betas.col(i) = xtwx_inv * xtwy;
                mat ci = xtwx_inv * xtw;
                mat si = x(i) * ci;
                S.row(i) = si;
            } catch (exception e) {
                emit error(e.what());
            }
        }
    }
    else
    {
        for (int i = 0; i < nDp & !checkCanceled(); i++)
        {
            vec w = mSpatialWeights[var].weightVector(i);
            mat xtw = trans(x % w);
            mat xtwx = xtw * x;
            mat xtwy = xtw * y;
            try
            {
                mat xtwx_inv = inv_sympd(xtwx);
                betas.col(i) = xtwx_inv * xtwy;
            } catch (exception e) {
                emit error(e.what());
            }
        }
    }
    return betas.t();
}
#ifdef ENABLE_OpenMP
vec GwmMultiscaleGWRAlgorithm::regressionVarOmp(const vec &x, const vec &y, const int var, mat &S)
{
    int nDp = x.n_rows;
    mat betas(1, nDp, fill::zeros);
    if (mHasHatMatrix)
    {
        mat ci, si;
        S = mat(mHasHatMatrix ? nDp : 1, nDp, fill::zeros);
#pragma omp parallel for num_threads(mOmpThreadNum)
        for (int i = 0; i < nDp; i++)
        {
            if(!checkCanceled())
            {
                vec w = mSpatialWeights[var].weightVector(i);
                mat xtw = trans(x % w);
                mat xtwx = xtw * x;
                mat xtwy = xtw * y;
                try
                {
                    mat xtwx_inv = inv_sympd(xtwx);
                    betas.col(i) = xtwx_inv * xtwy;
                    mat ci = xtwx_inv * xtw;
                    mat si = x(i) * ci;
                    S.row(i) = si;
                } catch (exception e) {
                    emit error(e.what());
                }
            }
        }
    }
    else
    {
#pragma omp parallel for num_threads(mOmpThreadNum)
        for (int i = 0; i < nDp; i++)
        {
            if(!checkCanceled())
            {
                vec w = mSpatialWeights[var].weightVector(i);
                mat xtw = trans(x % w);
                mat xtwx = xtw * x;
                mat xtwy = xtw * y;
                try
                {
                    mat xtwx_inv = inv_sympd(xtwx);
                    betas.col(i) = xtwx_inv * xtwy;
                } catch (exception e) {
                    emit error(e.what());
                }
            }
        }
    }
    return betas.t();
}
#endif
double GwmMultiscaleGWRAlgorithm::mBandwidthSizeCriterionAllCVSerial(GwmBandwidthWeight *bandwidthWeight)
{
    uword nDp = mDataPoints.n_rows;
    vec shat(2, fill::zeros);
    double cv = 0.0;
    for (uword i = 0; i < nDp & !checkCanceled(); i++)
    {
        vec d = mInitSpatialWeight.distance()->distance(i);
        vec w = bandwidthWeight->weight(d);
        w(i) = 0.0;
        mat xtw = trans(mX.each_col() % w);
        mat xtwx = xtw * mX;
        mat xtwy = xtw * mY;
        try
        {
            mat xtwx_inv = inv_sympd(xtwx);
            vec beta = xtwx_inv * xtwy;
            double res = mY(i) - det(mX.row(i) * beta);
            cv += res * res;
        }
        catch (...)
        {
            return DBL_MAX;
        }
    }
//    QString msg = QString(tr("%1 bandwidth: %2 (CV Score: %3)"))
//            .arg(bandwidthWeight->adaptive() ? "Adaptive" : "Fixed")
//            .arg(bandwidthWeight->bandwidth())
//            .arg(cv);
//    emit message(msg);
    if(!checkCanceled()) return cv;
    else return DBL_MAX;
}
#ifdef ENABLE_OpenMP
double GwmMultiscaleGWRAlgorithm::mBandwidthSizeCriterionAllCVOmp(GwmBandwidthWeight *bandwidthWeight)
{
    int nDp = mDataPoints.n_rows;
    vec shat(2, fill::zeros);
    vec cv_all(mOmpThreadNum, fill::zeros);
    bool flag = true;
#pragma omp parallel for num_threads(mOmpThreadNum)
    for (int i = 0; i < nDp; i++)
    {
        if (flag && !checkCanceled())
        {
            int thread = omp_get_thread_num();
            vec d = mInitSpatialWeight.distance()->distance(i);
            vec w = bandwidthWeight->weight(d);
            w(i) = 0.0;
            mat xtw = trans(mX.each_col() % w);
            mat xtwx = xtw * mX;
            mat xtwy = xtw * mY;
            try
            {
                mat xtwx_inv = inv_sympd(xtwx);
                vec beta = xtwx_inv * xtwy;
                double res = mY(i) - det(mX.row(i) * beta);
                cv_all(thread) += res * res;
            }
            catch (...)
            {
                flag = false;
            }
        }
    }
//    QString msg = QString(tr("%1 bandwidth: %2 (CV Score: %3)"))
//            .arg(bandwidthWeight->adaptive() ? "Adaptive" : "Fixed")
//            .arg(bandwidthWeight->bandwidth())
//            .arg(cv);
//    emit message(msg);
    if (flag && !checkCanceled())
    {
        return sum(cv_all);
    }
    else return DBL_MAX;
}
#endif
double GwmMultiscaleGWRAlgorithm::mBandwidthSizeCriterionAllAICSerial(GwmBandwidthWeight *bandwidthWeight)
{
    uword nDp = mDataPoints.n_rows, nVar = mIndepVars.size() + 1;
    mat betas(nVar, nDp, fill::zeros);
    vec shat(2, fill::zeros);
    for (uword i = 0; i < nDp & !checkCanceled(); i++)
    {
        vec d = mInitSpatialWeight.distance()->distance(i);
        vec w = bandwidthWeight->weight(d);
        mat xtw = trans(mX.each_col() % w);
        mat xtwx = xtw * mX;
        mat xtwy = xtw * mY;
        try
        {
            mat xtwx_inv = inv_sympd(xtwx);
            betas.col(i) = xtwx_inv * xtwy;
            mat ci = xtwx_inv * xtw;
            mat si = mX.row(i) * ci;
            shat(0) += si(0, i);
            shat(1) += det(si * si.t());
        }
        catch (std::exception e)
        {
            return DBL_MAX;
        }
    }
    double value = GwmMultiscaleGWRAlgorithm::AICc(mX, mY, betas.t(), shat);
//    QString msg = QString(tr("%1 bandwidth: %2 (AIC Score: %3)"))
//            .arg(bandwidthWeight->adaptive() ? "Adaptive" : "Fixed")
//            .arg(bandwidthWeight->bandwidth())
//            .arg(value);
//    emit message(msg);
    if(!checkCanceled()) return value;
    else return DBL_MAX;
}
#ifdef ENABLE_OpenMP
double GwmMultiscaleGWRAlgorithm::mBandwidthSizeCriterionAllAICOmp(GwmBandwidthWeight *bandwidthWeight)
{
    int nDp = mDataPoints.n_rows, nVar = mIndepVars.size() + 1;
    mat betas(nVar, nDp, fill::zeros);
    mat shat_all(2, mOmpThreadNum, fill::zeros);
    bool flag = true;
#pragma omp parallel for num_threads(mOmpThreadNum)
    for (int i = 0; i < nDp; i++)
    {
        if (flag && !checkCanceled())
        {
            int thread = omp_get_thread_num();
            vec d = mInitSpatialWeight.distance()->distance(i);
            vec w = bandwidthWeight->weight(d);
            mat xtw = trans(mX.each_col() % w);
            mat xtwx = xtw * mX;
            mat xtwy = xtw * mY;
            try
            {
                mat xtwx_inv = inv_sympd(xtwx);
                betas.col(i) = xtwx_inv * xtwy;
                mat ci = xtwx_inv * xtw;
                mat si = mX.row(i) * ci;
                shat_all(0, thread) += si(0, i);
                shat_all(1, thread) += det(si * si.t());
            }
            catch (...)
            {
                flag = false;
            }
        }
    }
    if (flag && !checkCanceled())
    {
        vec shat = sum(shat_all, 1);
        double value = GwmMultiscaleGWRAlgorithm::AICc(mX, mY, betas.t(), shat);
    //    QString msg = QString(tr("%1 bandwidth: %2 (AIC Score: %3)"))
    //            .arg(bandwidthWeight->adaptive() ? "Adaptive" : "Fixed")
    //            .arg(bandwidthWeight->bandwidth())
    //            .arg(value);
    //    emit message(msg);
        return value;
    }
    else return DBL_MAX;
}
#endif
double GwmMultiscaleGWRAlgorithm::mBandwidthSizeCriterionVarCVSerial(GwmBandwidthWeight *bandwidthWeight)
{
    int var = mBandwidthSelectionCurrentIndex;
    uword nDp = mDataPoints.n_rows;
    vec shat(2, fill::zeros);
    double cv = 0.0;
    for (uword i = 0; i < nDp & !checkCanceled(); i++)
    {
        vec d = mSpatialWeights[var].distance()->distance(i);
        vec w = bandwidthWeight->weight(d);
        w(i) = 0.0;
        mat xtw = trans(mXi % w);
        mat xtwx = xtw * mXi;
        mat xtwy = xtw * mYi;
        try
        {
            mat xtwx_inv = inv_sympd(xtwx);
            vec beta = xtwx_inv * xtwy;
            double res = mYi(i) - det(mXi(i) * beta);
            cv += res * res;
        }
        catch (...)
        {
            return DBL_MAX;
        }
    }
//    QString msg = QString(tr("%1 bandwidth: %2 (CV Score: %3)"))
//            .arg(bandwidthWeight->adaptive() ? "Adaptive" : "Fixed")
//            .arg(bandwidthWeight->bandwidth())
//            .arg(cv);
//    emit message(msg);
    if(!checkCanceled()) return cv;
    else return DBL_MAX;
}
#ifdef ENABLE_OpenMP
double GwmMultiscaleGWRAlgorithm::mBandwidthSizeCriterionVarCVOmp(GwmBandwidthWeight *bandwidthWeight)
{
    int var = mBandwidthSelectionCurrentIndex;
    int nDp = mDataPoints.n_rows;
    vec shat(2, fill::zeros);
    vec cv_all(mOmpThreadNum, fill::zeros);
    bool flag = true;
#pragma omp parallel for num_threads(mOmpThreadNum)
    for (int i = 0; i < nDp; i++)
    {
        if (flag && !checkCanceled())
        {
            int thread = omp_get_thread_num();
            vec d = mSpatialWeights[var].distance()->distance(i);
            vec w = bandwidthWeight->weight(d);
            w(i) = 0.0;
            mat xtw = trans(mXi % w);
            mat xtwx = xtw * mXi;
            mat xtwy = xtw * mYi;
            try
            {
                mat xtwx_inv = inv_sympd(xtwx);
                vec beta = xtwx_inv * xtwy;
                double res = mYi(i) - det(mXi(i) * beta);
                cv_all(thread) += res * res;
            }
            catch (...)
            {
                flag = false;
            }
        }
    }
    if (flag && !checkCanceled())
    {
//    QString msg = QString(tr("%1 bandwidth: %2 (CV Score: %3)"))
//            .arg(bandwidthWeight->adaptive() ? "Adaptive" : "Fixed")
//            .arg(bandwidthWeight->bandwidth())
//            .arg(cv);
//    emit message(msg);
        return sum(cv_all);
    }
    else return DBL_MAX;
}
#endif
double GwmMultiscaleGWRAlgorithm::mBandwidthSizeCriterionVarAICSerial(GwmBandwidthWeight *bandwidthWeight)
{
    int var = mBandwidthSelectionCurrentIndex;
    uword nDp = mDataPoints.n_rows, nVar = mIndepVars.size() + 1;
    mat betas(1, nDp, fill::zeros);
    vec shat(2, fill::zeros);
    for (uword i = 0; i < nDp & !checkCanceled(); i++)
    {
        vec d = mSpatialWeights[var].distance()->distance(i);
        vec w = bandwidthWeight->weight(d);
        mat xtw = trans(mXi % w);
        mat xtwx = xtw * mXi;
        mat xtwy = xtw * mYi;
        try
        {
            mat xtwx_inv = inv_sympd(xtwx);
            betas.col(i) = xtwx_inv * xtwy;
            mat ci = xtwx_inv * xtw;
            mat si = mXi(i) * ci;
            shat(0) += si(0, i);
            shat(1) += det(si * si.t());
        }
        catch (std::exception e)
        {
            return DBL_MAX;
        }
    }
    double value = GwmMultiscaleGWRAlgorithm::AICc(mXi, mYi, betas.t(), shat);
//    QString msg = QString(tr("%1 bandwidth: %2 (AIC Score: %3)"))
//            .arg(bandwidthWeight->adaptive() ? "Adaptive" : "Fixed")
//            .arg(bandwidthWeight->bandwidth())
//            .arg(value);
//    emit message(msg);
    if(!checkCanceled()) return value;
    else return DBL_MAX;
}
#ifdef ENABLE_OpenMP
double GwmMultiscaleGWRAlgorithm::mBandwidthSizeCriterionVarAICOmp(GwmBandwidthWeight *bandwidthWeight)
{
    int var = mBandwidthSelectionCurrentIndex;
    int nDp = mDataPoints.n_rows, nVar = mIndepVars.size() + 1;
    mat betas(1, nDp, fill::zeros);
    mat shat_all(2, mOmpThreadNum, fill::zeros);
    bool flag = true;
#pragma omp parallel for num_threads(mOmpThreadNum)
    for (int i = 0; i < nDp; i++)
    {
        if (flag && !checkCanceled())
        {
            int thread = omp_get_thread_num();
            vec d = mSpatialWeights[var].distance()->distance(i);
            vec w = bandwidthWeight->weight(d);
            mat xtw = trans(mXi % w);
            mat xtwx = xtw * mXi;
            mat xtwy = xtw * mYi;
            try
            {
                mat xtwx_inv = inv_sympd(xtwx);
                betas.col(i) = xtwx_inv * xtwy;
                mat ci = xtwx_inv * xtw;
                mat si = mXi(i) * ci;
                shat_all(0, thread) += si(0, i);
                shat_all(1, thread) += det(si * si.t());
            }
            catch (std::exception e)
            {
                flag = false;
            }
        }
    }
    if (flag && !checkCanceled())
    {
        vec shat = sum(shat_all, 1);
        double value = GwmMultiscaleGWRAlgorithm::AICc(mXi, mYi, betas.t(), shat);
    //    QString msg = QString(tr("%1 bandwidth: %2 (AIC Score: %3)"))
    //            .arg(bandwidthWeight->adaptive() ? "Adaptive" : "Fixed")
    //            .arg(bandwidthWeight->bandwidth())
    //            .arg(value);
    //    emit message(msg);
        return value;
    }
    return DBL_MAX;
}
#endif
GwmMultiscaleGWRAlgorithm::BandwidthSizeCriterionFunction GwmMultiscaleGWRAlgorithm::bandwidthSizeCriterionAll(GwmMultiscaleGWRAlgorithm::BandwidthSelectionCriterionType type)
{
    QMap<BandwidthSelectionCriterionType, QMap<IParallelalbe::ParallelType, BandwidthSizeCriterionFunction> > mapper = {
        std::make_pair<BandwidthSelectionCriterionType, QMap<IParallelalbe::ParallelType, BandwidthSizeCriterionFunction> >(BandwidthSelectionCriterionType::CV, {
            std::make_pair(IParallelalbe::ParallelType::SerialOnly, &GwmMultiscaleGWRAlgorithm::mBandwidthSizeCriterionAllCVSerial),
        #ifdef ENABLE_OpenMP
            std::make_pair(IParallelalbe::ParallelType::OpenMP, &GwmMultiscaleGWRAlgorithm::mBandwidthSizeCriterionAllCVOmp),
        #endif
            std::make_pair(IParallelalbe::ParallelType::CUDA, &GwmMultiscaleGWRAlgorithm::mBandwidthSizeCriterionAllCVSerial)
        }),
        std::make_pair<BandwidthSelectionCriterionType, QMap<IParallelalbe::ParallelType, BandwidthSizeCriterionFunction> >(BandwidthSelectionCriterionType::AIC, {
            std::make_pair(IParallelalbe::ParallelType::SerialOnly, &GwmMultiscaleGWRAlgorithm::mBandwidthSizeCriterionAllAICSerial),
        #ifdef ENABLE_OpenMP
            std::make_pair(IParallelalbe::ParallelType::OpenMP, &GwmMultiscaleGWRAlgorithm::mBandwidthSizeCriterionAllAICOmp),
        #endif
            std::make_pair(IParallelalbe::ParallelType::CUDA, &GwmMultiscaleGWRAlgorithm::mBandwidthSizeCriterionAllAICSerial)
        })
    };
    return mapper[type][mParallelType];
}

GwmMultiscaleGWRAlgorithm::BandwidthSizeCriterionFunction GwmMultiscaleGWRAlgorithm::bandwidthSizeCriterionVar(GwmMultiscaleGWRAlgorithm::BandwidthSelectionCriterionType type)
{
    QMap<BandwidthSelectionCriterionType, QMap<IParallelalbe::ParallelType, BandwidthSizeCriterionFunction> > mapper = {
        std::make_pair<BandwidthSelectionCriterionType, QMap<IParallelalbe::ParallelType, BandwidthSizeCriterionFunction> >(BandwidthSelectionCriterionType::CV, {
            std::make_pair(IParallelalbe::ParallelType::SerialOnly, &GwmMultiscaleGWRAlgorithm::mBandwidthSizeCriterionVarCVSerial),
        #ifdef ENABLE_OpenMP
            std::make_pair(IParallelalbe::ParallelType::OpenMP, &GwmMultiscaleGWRAlgorithm::mBandwidthSizeCriterionVarCVOmp),
        #endif
            std::make_pair(IParallelalbe::ParallelType::CUDA, &GwmMultiscaleGWRAlgorithm::mBandwidthSizeCriterionVarCVSerial)
        }),
        std::make_pair<BandwidthSelectionCriterionType, QMap<IParallelalbe::ParallelType, BandwidthSizeCriterionFunction> >(BandwidthSelectionCriterionType::AIC, {
            std::make_pair(IParallelalbe::ParallelType::SerialOnly, &GwmMultiscaleGWRAlgorithm::mBandwidthSizeCriterionVarAICSerial),
        #ifdef ENABLE_OpenMP
            std::make_pair(IParallelalbe::ParallelType::OpenMP, &GwmMultiscaleGWRAlgorithm::mBandwidthSizeCriterionVarAICOmp),
        #endif
            std::make_pair(IParallelalbe::ParallelType::CUDA, &GwmMultiscaleGWRAlgorithm::mBandwidthSizeCriterionVarAICSerial)
        })
    };
    return mapper[type][mParallelType];
}

void GwmMultiscaleGWRAlgorithm::setParallelType(const IParallelalbe::ParallelType &type)
{
    if (parallelAbility() & type)
    {
        mParallelType = type;
        switch (type) {
        case IParallelalbe::ParallelType::SerialOnly:
            mRegressionAll = &GwmMultiscaleGWRAlgorithm::regressionAllSerial;
            mRegressionVar = &GwmMultiscaleGWRAlgorithm::regressionVarSerial;
            break;
#ifdef ENABLE_OpenMP
        case IParallelalbe::ParallelType::OpenMP:
            mRegressionAll = &GwmMultiscaleGWRAlgorithm::regressionAllOmp;
            mRegressionVar = &GwmMultiscaleGWRAlgorithm::regressionVarOmp;
            break;
#endif
//        case IParallelalbe::ParallelType::CUDA:
//            mRegressionAll = &GwmMultiscaleGWRAlgorithm::regressionAllOmp;
//            mRegressionVar = &GwmMultiscaleGWRAlgorithm::regressionVarOmp;
//            break;
        default:
            break;
        }
    }
}

void GwmMultiscaleGWRAlgorithm::setSpatialWeights(const QList<GwmSpatialWeight> &spatialWeights)
{
    GwmSpatialMultiscaleAlgorithm::setSpatialWeights(spatialWeights);
    if (spatialWeights.size() > 0)
    {
        mInitSpatialWeight = spatialWeights[0];
    }
}
