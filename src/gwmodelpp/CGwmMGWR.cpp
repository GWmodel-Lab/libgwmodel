#include "CGwmMGWR.h"
#ifdef ENABLE_OPENMP
#include <omp.h>
#endif
#include <exception>
#include "gwmodel.h"
#include <spatialweight/CGwmCRSDistance.h>
#include "CGwmBandwidthSelector.h"
#include "CGwmVariableForwardSelector.h"
#include <assert.h>

using namespace std;

int CGwmMGWR::treeChildCount = 0;

unordered_map<CGwmMGWR::BandwidthInitilizeType,string> CGwmMGWR::BandwidthInitilizeTypeNameMapper = {
    make_pair(CGwmMGWR::BandwidthInitilizeType::Null, ("Not initilized, not specified")),
    make_pair(CGwmMGWR::BandwidthInitilizeType::Initial, ("Initilized")),
    make_pair(CGwmMGWR::BandwidthInitilizeType::Specified, ("Specified"))
};

unordered_map<CGwmMGWR::BandwidthSelectionCriterionType,string> CGwmMGWR::BandwidthSelectionCriterionTypeNameMapper = {
    make_pair(CGwmMGWR::BandwidthSelectionCriterionType::CV, ("CV")),
    make_pair(CGwmMGWR::BandwidthSelectionCriterionType::AIC, ("AIC"))
};

unordered_map<CGwmMGWR::BackFittingCriterionType,string> CGwmMGWR::BackFittingCriterionTypeNameMapper = {
    make_pair(CGwmMGWR::BackFittingCriterionType::CVR, ("CVR")),
    make_pair(CGwmMGWR::BackFittingCriterionType::dCVR, ("dCVR"))
};

GwmRegressionDiagnostic CGwmMGWR::CalcDiagnostic(const mat& x, const vec& y, const mat& betas, const vec& shat)
{//诊断值
    vec r = y - sum(betas % x, 1);
    double rss = sum(r % r);
    double n = (double)x.n_rows;
    double AIC = n * log(rss / n) + n * log(2 * datum::pi) + n + shat(0);
    double AICc = n * log(rss / n) + n * log(2 * datum::pi) + n * ((n + shat(0)) / (n - 2 - shat(0)));
    double edf = n - 2 * shat(0) + shat(1);
    double enp = 2 * shat(0) - shat(1);
    double yss = sum((y - mean(y)) % (y - mean(y)));
    double r2 = 1 - rss / yss;
    double r2_adj = 1 - (1 - r2) * (n - 1) / (edf - 1);
    return { rss, AIC, AICc, enp, edf, r2, r2_adj };
    /*// 诊断信息
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
    GwmRegressionDiagnostic diagnostic;
    diagnostic.RSS = RSSg;
    diagnostic.AICc = AICc;
    diagnostic.ENP = enp;
    diagnostic.EDF = edf;
    diagnostic.RSquareAdjust = adjustRsquare;
    diagnostic.RSquare = Rsquare;
    return diagnostic;*/
}

CGwmMGWR::CGwmMGWR()
    : CGwmSpatialMultiscaleAlgorithm()
{
}
/*
void CGwmMGWR::setCanceled(bool canceled)
{
    selector.setCanceled(canceled);
    return CGwmTaskThread::setCanceled(canceled);
}
*/
//OLS计算代码
/*
GwmBasicGWRAlgorithm::OLSVar CGwmMGWR::CalOLS(const mat &x, const vec &y){
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
*/
void CGwmMGWR::run()
{
    createRegressionDistanceParameter();
    //assert(mRegressionDistanceParameter != nullptr);
    //initPoints();
    setXY(mX, mY, mSourceLayer, mDepVar, mIndepVars);
    uword nDp = mX.n_rows, nVar = mX.n_cols;

    // ********************************
    // Centering and scaling predictors
    // ********************************
    mX0 = mX;
    mY0 = mY;
    for (uword i = 1; i < nVar ; i++)
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
    for (uword i = 0; i < nVar ; i++)
    {
        if (mBandwidthInitilize[i] == BandwidthInitilizeType::Null)
        {
            //emit message(("Calculating the initial bandwidth for %1 ...").arg(i == 0 ? "Intercept" : mIndepVars[i-1].name));
            mBandwidthSizeCriterion = bandwidthSizeCriterionVar(mBandwidthSelectionApproach[i]);
            mBandwidthSelectionCurrentIndex = i;
            mXi = mX.col(i);
            CGwmBandwidthWeight* bw0 = bandwidth(i);
            bool adaptive = bw0->adaptive();
            mselector.setBandwidth(bw0);
            mselector.setLower(adaptive ? mAdaptiveLower : 0.0);
            mselector.setUpper(adaptive ? mSourceLayer->featureCount() : mSpatialWeights[i].distance()->maxDistance());
            CGwmBandwidthWeight* bw = mselector.optimize(this);
            if (bw)
            {
                mSpatialWeights[i].setWeight(bw);
            }
        }
    }
    /*
    if(mOLS&&!checkCanceled()){
        mOLSVar = CalOLS(mX,mY);
    }
    */
    // *****************************************************
    // Calculate the initial beta0 from the above bandwidths
    // *****************************************************
    //emit message(tr("Calculating the initial beta0 from the above bandwidths ..."));
    CGwmBandwidthWeight* bw0 = bandwidth(0);
    bool adaptive = bw0->adaptive();
    mBandwidthSizeCriterion = bandwidthSizeCriterionAll(mBandwidthSelectionApproach[0]);
    CGwmBandwidthSelector initBwSelector;
    initBwSelector.setBandwidth(bw0);
    initBwSelector.setLower(adaptive ? mAdaptiveLower : 0.0);
    initBwSelector.setUpper(adaptive ? mSourceLayer->featureCount() : mSpatialWeights[0].distance()->maxDistance());
    CGwmBandwidthWeight* initBw = initBwSelector.optimize(this);
    if (!initBw)
    {
        //emit error(tr("Cannot select initial bandwidth."));
        return;
    }
    mInitSpatialWeight.setWeight(initBw);

    // 初始化诊断信息矩阵
    if (mHasHatMatrix )
    {
        mS0 = mat(nDp, nDp, fill::zeros);
        mSArray = cube(nDp, nDp, nVar, fill::zeros);
        mC = cube(nVar, nDp, nDp, fill::zeros);
    }

    mBetas = regression(mX, mY);

    if (mHasHatMatrix )
    {
        mat betasSE, S;
        vec shat, qdiag;
        mBetas = regressionHatmatrix(mX, mY, betasSE, shat, qdiag, S);
        mDiagnostic = CalcDiagnostic(mX, mY, mBetas, shat);
        double trS = shat(0), trStS = shat(1);
        double sigmaHat = mDiagnostic.RSS / (nDp - 2 * trS + trStS);
        vec yhat = Fitted(mX, mBetas);
        vec residual = mY - yhat;
        vec stu_res = residual / sqrt(sigmaHat * qdiag);
        mBetasTV = mBetas / mBetasSE;
        vec dybar2 = (mY - mean(mY)) % (mY - mean(mY));
        vec dyhat2 = (mY - yhat) % (mY - yhat);
        vec localR2 = vec(nDp, nVar,fill::zeros);
        for (uword i = 0; i < nDp; i++)
        {
            for(uword j=0;j<nVar;j++){
                vec w = mSpatialWeights[j].weightVector(i);
                double tss = sum(dybar2 % w);
                double rss = sum(dyhat2 % w);
                localR2(i,j) = (tss - rss) / tss;
            }
        }
        createResultLayer({//结果及诊断信息
            make_tuple(string("%1"), mBetas, NameFormat::VarName),
            make_tuple(string("y"), mY, NameFormat::Fixed),
            make_tuple(string("yhat"), yhat, NameFormat::Fixed),
            make_tuple(string("residual"), residual, NameFormat::Fixed),
            make_tuple(string("Stud_residual"), stu_res, NameFormat::Fixed),
            make_tuple(string("SE"), mBetasSE, NameFormat::PrefixVarName),
            make_tuple(string("TV"), mBetasTV, NameFormat::PrefixVarName),
            make_tuple(string("localR2"), localR2, NameFormat::Fixed)
            /*qMakePair(QString("%1"), mBetas),
            qMakePair(QString("yhat"), yhat),
            qMakePair(QString("residual"), residual),
            qMakePair(QString("%1_SE"),mBetasSE),
            qMakePair(QString("%1_TV"),mBetasTV)*/
        });
    }
    else
    {
        createResultLayer({
            make_tuple(string("%1"), mBetas, NameFormat::VarName)
            //qMakePair(QString("%1"), mBetas)
        });
    }

    /*if(!checkCanceled())
    {
        //emit success();
        //emit tick(100,100);
    }
    else return;*/
}

void CGwmMGWR::createRegressionDistanceParameter()
{//回归距离计算
    for (uword i = 0; i < mIndepVars.size(); i++){
        if (mSpatialWeights[i].distance()->type() == CGwmDistance::DistanceType::CRSDistance || 
            mSpatialWeights[i].distance()->type() == CGwmDistance::DistanceType::MinkwoskiDistance)
        {
            mSpatialWeights[i].distance()->makeParameter({
                mSourceLayer->points(),
                mSourceLayer->points()
            });
        }
    }
}

mat CGwmMGWR::regression(const mat &x, const vec &y)
{
    uword nDp = x.n_rows, nVar = x.n_cols;
    mat betas = (this->*mRegressionAll)(x, y);

    if (mHasHatMatrix)
    {
        mat idm(nVar, nVar, fill::eye);
        for (uword i = 0; i < nVar; ++i)
        {
            for (uword j = 0; j < nDp ; ++j)
            {
                mSArray.slice(i).row(j) = x(j, i) * (idm.row(i) * mC.slice(j));
            }
        }
    }

    // ***********************************************************
    // Select the optimum bandwidths for each independent variable
    // ***********************************************************
    //emit message(QString("-------- Select the Optimum Bandwidths for each Independent Varialbe --------"));
    uvec bwChangeNo(nVar, fill::zeros);
    vec resid = y - Fitted(x, betas);
    double RSS0 = sum(resid % resid), RSS1 = DBL_MAX;
    double criterion = DBL_MAX;
    for (int iteration = 1; iteration <= mMaxIteration && criterion > mCriterionThreshold; iteration++)
    {
        //emit tick(iteration - 1, mMaxIteration);
        for (uword i = 0; i < nVar  ; i++)
        {
            //QString varName = i == 0 ? QStringLiteral("Intercept") : mIndepVars[i-1].name;
            vec fi = betas.col(i) % x.col(i);
            vec yi = resid + fi;
            if (mBandwidthInitilize[i] != BandwidthInitilizeType::Specified)
            {
                //emit message(QString("Now select an optimum bandwidth for the variable: %1").arg(varName));
                mBandwidthSizeCriterion = bandwidthSizeCriterionVar(mBandwidthSelectionApproach[i]);
                mBandwidthSelectionCurrentIndex = i;
                mYi = yi;
                mXi = mX.col(i);
                CGwmBandwidthWeight* bwi0 = bandwidth(i);
                bool adaptive = bwi0->adaptive();
                CGwmBandwidthSelector mselector;
                mselector.setBandwidth(bwi0);
                mselector.setLower(adaptive ? mAdaptiveLower : 0.0);
                mselector.setUpper(adaptive ? mSourceLayer->featureCount() : mSpatialWeights[i].distance()->maxDistance());
                CGwmBandwidthWeight* bwi = mselector.optimize(this);
                double bwi0s = bwi0->bandwidth(), bwi1s = bwi->bandwidth();
                //emit message(QString("The newly selected bandwidth for variable %1 is %2 (last is %3, difference is %4)")
                            // .arg(varName).arg(bwi1s).arg(bwi0s).arg(abs(bwi1s - bwi0s)));
                if (abs(bwi1s - bwi0s) > mBandwidthSelectThreshold[i])
                {
                    bwChangeNo(i) = 0;
                    //emit message(QString("The bandwidth for variable %1 will be continually selected in the next iteration").arg(varName));
                }
                else
                {
                    bwChangeNo(i) += 1;
                    if (bwChangeNo(i) >= mBandwidthSelectRetryTimes)
                    {
                        mBandwidthInitilize[i] = BandwidthInitilizeType::Specified;
                        //emit message(QString("The bandwidth for variable %1 seems to be converged and will be kept the same in the following iterations.").arg(varName));
                    }
                    else
                    {
                        //emit message(QString("The bandwidth for variable %1 seems to be converged for %2 times. It will be continually optimized in the next %3 times.")
                                     //.arg(varName).arg(bwChangeNo(i)).arg(mBandwidthSelectRetryTimes - bwChangeNo(i)));
                    }
                }
                mSpatialWeights[i].setWeight(bwi);
            }

            mat S;
            betas.col(i) = (this->*mRegressionVar)(x.col(i), yi, i, S);
            if (mHasHatMatrix )
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
        //QString criterionName = mCriterionType == BackFittingCriterionType::CVR ? "change value of RSS (CVR)" : "differential change value of RSS (dCVR)";
        //emit message(QString("Iteration %1 the %2 is %3").arg(iteration).arg(criterionName).arg(criterion));
        RSS0 = RSS1;
        //emit message(QString("---- End of Iteration %1 ----").arg(iteration));
    }
    //emit message(QString("-------- [End] Select the Optimum Bandwidths for each Independent Varialbe --------"));
    mRSS0 = RSS0;
    return betas;
}
mat CGwmMGWR::regressionHatmatrixSerial(const mat& x, const vec& y, mat& betasSE, vec& shat, vec& qdiag, mat& S)
{
    uword nDp = mSourceLayer->featureCount(), nVar = mIndepVars.size() + 1;
    mat betas(nVar, nDp, arma::fill::zeros);
    betasSE = mat(nVar, nDp, arma::fill::zeros);
    qdiag = vec(nDp, arma::fill::zeros);
    S = mat(isStoreS() ? nDp : 1, nDp, arma::fill::zeros);
    mat rowsumSE(nDp, 1, arma::fill::ones);
    vec s_hat1(nDp, arma::fill::zeros), s_hat2(nDp, arma::fill::zeros);
    for (size_t i = 0; i < nDp; i++)
    {
        vec w(nDp, arma::fill::ones);
        for (auto&& sw : mSpatialWeights)
        {
            vec w_m = sw.weightVector(i);
            w = w % w_m;
        }
        mat ws(1, nVar, arma::fill::ones);
        mat xtw = trans(x %(w * ws));
        mat xtwx = xtw * x;
        mat xtwy = trans(x) * (w % y);
        try
        {
            mat xtwx_inv = xtwx.i();
            betas.col(i) = xtwx_inv * xtwy;
            // hatmatrix
            mat ci = xtwx_inv * xtw;
            mat si = x.row(i) * ci;
            betasSE.col(i) = (ci % ci) * rowsumSE;
            s_hat1(i) = si(0, i);
            s_hat2(i) = det(si * si.t());
            mat onei(1, nDp, arma::fill::zeros);
            onei(i) = 1;
            mat p = (onei - si).t();
            qdiag += p % p;
            S.row(isStoreS() ? i : 0) = si;
        }
        catch(const std::exception& e)
        {
            std::cerr << e.what() << '\n';
        }
    }
    shat = {sum(s_hat1), sum(s_hat2)};
    betasSE = betasSE.t();
    return betas.t();
}
bool CGwmMGWR::isValid()
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
        CGwmBandwidthWeight* bw = mSpatialWeights[i].weight<CGwmBandwidthWeight>();
        if (mBandwidthInitilize[i] == CGwmMGWR::Specified || mBandwidthInitilize[i] == CGwmMGWR::Initial)
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
/*
void CGwmMGWR::initPoints()
{
    int nDp = mSourceLayer->featureCount();
    mDataPoints = mat(nDp, 2, fill::zeros);
    //QgsFeatureIterator iterator = mSourceLayer->getFeatures();
    //QgsFeature f;
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
    if (mInitSpatialWeight.distance()->type() == CGwmDistance::CRSDistance || mInitSpatialWeight.distance()->type() == CGwmDistance::MinkwoskiDistance)
    {
        CGwmCRSDistance* d = mInitSpatialWeight.distance<CGwmCRSDistance>();
        d->setDataPoints(&mDataPoints);
        d->setFocusPoints(&mRegressionPoints);
    }
    for (const CGwmSpatialWeight& sw : mSpatialWeights)
    {
        if (sw.distance()->type() == CGwmDistance::CRSDistance || sw.distance()->type() == CGwmDistance::MinkwoskiDistance)
        {
            CGwmCRSDistance* d = sw.distance<CGwmCRSDistance>();
            d->setDataPoints(&mDataPoints);
            d->setFocusPoints(&mRegressionPoints);
        }
    }
}
*/
/*
void CGwmMGWR::initXY(mat &x, mat &y, const GwmVariable &depVar, const vector<GwmVariable> &indepVars)
{//
    int nDp = mSourceLayer->featureCount(), nVar = indepVars.size() + 1;
    // Data layer and X,Y
    x = mat(nDp, nVar, arma::fill::zeros);
    y = vec(nDp, arma::fill::zeros);
    mat data = mSourceLayer->data();
    bool ok = false;
    for (uword i = 0; i<nVar; i++)
    {
        double vY = f.attribute(depVar.name).toDouble(&ok);
        //if (ok)
        {
            y(i) = vY;
            x(i, 0) = 1.0;
            for (int k = 0; k < indepVars.size(); k++)
            {
                double vX = f.attribute(indepVars[k].name).toDouble(&ok);
                //if (ok) x(i, k + 1) = vX;
                //else emit error(tr("Independent variable value cannot convert to a number. Set to 0."));
            }
        }
        //else emit error(tr("Dependent variable value cannot convert to a number. Set to 0."));
    }
}
*/
void CGwmMGWR::setXY(mat& x, mat& y, const CGwmSimpleLayer* layer, const GwmVariable& depVar, const vector<GwmVariable>& indepVars)
{
    uword nDp = layer->featureCount(), nVar = indepVars.size() + 1;
    arma::uvec indepVarIndeces(indepVars.size());
    for (size_t i = 0; i < indepVars.size(); i++)
    {
        assert(indepVars[i].index < layer->data().n_cols);
        indepVarIndeces(i) = indepVars[i].index;
    }
    x = join_rows(mat(nDp, 1, arma::fill::ones), layer->data().cols(indepVarIndeces));
    y = layer->data().col(depVar.index);
}
void CGwmMGWR::createResultLayer(initializer_list<CreateResultLayerDataItem> items)
{//输出结果图层
    mat layerPoints = mSourceLayer->points();
    uword layerFeatureCount = layerPoints.n_rows;
    mat layerData(layerFeatureCount, 0);
    vector<string> layerFields;
    for (auto &&i : items)
    {
        NameFormat nf = get<2>(i);
        mat column = get<1>(i);
        string name = get<0>(i);
        if (nf == NameFormat::Fixed)
        {
            layerData = join_rows(layerData, column.col(0));
            layerFields.push_back(name);
        }
        else
        {
            layerData = join_rows(layerData, column);
            for (size_t k = 0; k < column.n_cols; k++)
            {
                string variableName = k == 0 ? "Intercept" : mIndepVars[k - 1].name;
                string fieldName;
                switch (nf)
                {
                case NameFormat::VarName:
                    fieldName = variableName;
                    break;
                case NameFormat::PrefixVarName:
                    fieldName = variableName + "_" + name;
                    break;
                case NameFormat::SuffixVariable:
                    fieldName = name + "_" + variableName;
                    break;
                default:
                    fieldName = variableName;
                }
                layerFields.push_back(fieldName);
            }
        }
        
    }
    
    mResultLayer = new CGwmSimpleLayer(layerPoints, layerData, layerFields);
    /*//emit message("Creating result layer...");
    QgsVectorLayer* srcLayer = mRegressionLayer ? mRegressionLayer : mSourceLayer;
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
    mResultLayer->commitChanges();*/
}


mat CGwmMGWR::regressionAllSerial(const mat& x, const vec& y)
{
    int nDp = x.n_rows, nVar = x.n_cols;
    mat betas(nVar, nDp, fill::zeros);
    if (mHasHatMatrix )
    {
        mat betasSE(nVar, nDp, fill::zeros);
        for (int i = 0; i < nDp ; i++)
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
                //emit error(e.what());
            }
        }
        mBetasSE = betasSE.t();
    }
    else
    {
        for (int i = 0; i < nDp ; i++)
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
                //emit error(e.what());
            }
        }
    }
    return betas.t();
}
#ifdef ENABLE_OPENMP
mat CGwmMGWR::regressionAllOmp(const mat &x, const vec &y)
{
    int nDp = x.n_rows, nVar = x.n_cols;
    mat betas(nVar, nDp, fill::zeros);
    if (mHasHatMatrix )
    {
        mat betasSE(nVar, nDp, fill::zeros);
#pragma omp parallel for num_threads(mOmpThreadNum)
        for (int i = 0; i < nDp; i++)
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
                    //emit error(e.what());
                }
        }
        mBetasSE = betasSE.t();
    }
    else
    {
#pragma omp parallel for num_threads(mOmpThreadNum)
        for (int i = 0; i < nDp; i++)
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
                    //emit error(e.what());
                }
        }
    }
    return betas.t();
}
#endif
vec CGwmMGWR::regressionVarSerial(const vec &x, const vec &y, const int var, mat &S)
{
    int nDp = x.n_rows;
    mat betas(1, nDp, fill::zeros);
    if (mHasHatMatrix )
    {
        mat ci, si;
        S = mat(mHasHatMatrix ? nDp : 1, nDp, fill::zeros);
        for (int i = 0; i < nDp  ; i++)
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
                //emit error(e.what());
            }
        }
    }
    else
    {
        for (int i = 0; i < nDp  ; i++)
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
                //emit error(e.what());
            }
        }
    }
    return betas.t();
}
#ifdef ENABLE_OPENMP
vec CGwmMGWR::regressionVarOmp(const vec &x, const vec &y, const int var, mat &S)
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
                    //emit error(e.what());
                }
        }
    }
    else
    {
#pragma omp parallel for num_threads(mOmpThreadNum)
        for (int i = 0; i < nDp; i++)
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
                    //emit error(e.what());
                }
        }
    }
    return betas.t();
}
#endif
double CGwmMGWR::mBandwidthSizeCriterionAllCVSerial(CGwmBandwidthWeight *bandwidthWeight)
{
    uword nDp = mDataPoints.n_rows;
    vec shat(2, fill::zeros);
    double cv = 0.0;
    for (uword i = 0; i < nDp; i++)
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
    return cv;
    //else return DBL_MAX;
}
#ifdef ENABLE_OPENMP
double CGwmMGWR::mBandwidthSizeCriterionAllCVOmp(CGwmBandwidthWeight *bandwidthWeight)
{
    int nDp = mDataPoints.n_rows;
    vec shat(2, fill::zeros);
    vec cv_all(mOmpThreadNum, fill::zeros);
    bool flag = true;
#pragma omp parallel for num_threads(mOmpThreadNum)
    for (int i = 0; i < nDp; i++)
    {
        if (flag)
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
    if (flag)
    {
        return sum(cv_all);
    }
    else return DBL_MAX;
}
#endif
double CGwmMGWR::mBandwidthSizeCriterionAllAICSerial(CGwmBandwidthWeight *bandwidthWeight)
{
    uword nDp = mDataPoints.n_rows, nVar = mIndepVars.size() + 1;
    mat betas(nVar, nDp, fill::zeros);
    vec shat(2, fill::zeros);
    for (uword i = 0; i < nDp ; i++)
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
    double value = CGwmMGWR::AICc(mX, mY, betas.t(), shat);
//    QString msg = QString(tr("%1 bandwidth: %2 (AIC Score: %3)"))
//            .arg(bandwidthWeight->adaptive() ? "Adaptive" : "Fixed")
//            .arg(bandwidthWeight->bandwidth())
//            .arg(value);
//    emit message(msg);
    return value;
    //else return DBL_MAX;
}
#ifdef ENABLE_OPENMP
double CGwmMGWR::mBandwidthSizeCriterionAllAICOmp(CGwmBandwidthWeight *bandwidthWeight)
{
    int nDp = mDataPoints.n_rows, nVar = mIndepVars.size() + 1;
    mat betas(nVar, nDp, fill::zeros);
    mat shat_all(2, mOmpThreadNum, fill::zeros);
    bool flag = true;
#pragma omp parallel for num_threads(mOmpThreadNum)
    for (int i = 0; i < nDp; i++)
    {
        if (flag)
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
    if (flag)
    {
        vec shat = sum(shat_all, 1);
        double value = CGwmMGWR::AICc(mX, mY, betas.t(), shat);
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
double CGwmMGWR::mBandwidthSizeCriterionVarCVSerial(CGwmBandwidthWeight *bandwidthWeight)
{
    int var = mBandwidthSelectionCurrentIndex;
    uword nDp = mDataPoints.n_rows;
    vec shat(2, fill::zeros);
    double cv = 0.0;
    for (uword i = 0; i < nDp; i++)
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
    return cv;
    //else return DBL_MAX;
}
#ifdef ENABLE_OPENMP
double CGwmMGWR::mBandwidthSizeCriterionVarCVOmp(CGwmBandwidthWeight *bandwidthWeight)
{
    int var = mBandwidthSelectionCurrentIndex;
    int nDp = mDataPoints.n_rows;
    vec shat(2, fill::zeros);
    vec cv_all(mOmpThreadNum, fill::zeros);
    bool flag = true;
#pragma omp parallel for num_threads(mOmpThreadNum)
    for (int i = 0; i < nDp; i++)
    {
        if (flag)
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
    if (flag)
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
double CGwmMGWR::mBandwidthSizeCriterionVarAICSerial(CGwmBandwidthWeight *bandwidthWeight)
{
    int var = mBandwidthSelectionCurrentIndex;
    uword nDp = mDataPoints.n_rows, nVar = mIndepVars.size() + 1;
    mat betas(1, nDp, fill::zeros);
    vec shat(2, fill::zeros);
    for (uword i = 0; i < nDp ; i++)
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
    double value = CGwmMGWR::AICc(mXi, mYi, betas.t(), shat);
//    QString msg = QString(tr("%1 bandwidth: %2 (AIC Score: %3)"))
//            .arg(bandwidthWeight->adaptive() ? "Adaptive" : "Fixed")
//            .arg(bandwidthWeight->bandwidth())
//            .arg(value);
//    emit message(msg);
    return value;
    //else return DBL_MAX;
}
#ifdef ENABLE_OPENMP
double CGwmMGWR::mBandwidthSizeCriterionVarAICOmp(CGwmBandwidthWeight *bandwidthWeight)
{
    int var = mBandwidthSelectionCurrentIndex;
    int nDp = mDataPoints.n_rows, nVar = mIndepVars.size() + 1;
    mat betas(1, nDp, fill::zeros);
    mat shat_all(2, mOmpThreadNum, fill::zeros);
    bool flag = true;
#pragma omp parallel for num_threads(mOmpThreadNum)
    for (int i = 0; i < nDp; i++)
    {
        if (flag)
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
    if (flag)
    {
        vec shat = sum(shat_all, 1);
        double value = CGwmMGWR::AICc(mXi, mYi, betas.t(), shat);
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

CGwmMGWR::BandwidthSizeCriterionFunction CGwmMGWR::bandwidthSizeCriterionAll(CGwmMGWR::BandwidthSelectionCriterionType type)
{
    unordered_map<BandwidthSelectionCriterionType, unordered_map<ParallelType, BandwidthSizeCriterionFunction> > mapper = {
        std::make_pair<BandwidthSelectionCriterionType, unordered_map<ParallelType, BandwidthSizeCriterionFunction> >(BandwidthSelectionCriterionType::CV, {
            std::make_pair(ParallelType::SerialOnly, &CGwmMGWR::mBandwidthSizeCriterionAllCVSerial),
        #ifdef ENABLE_OPENMP
            std::make_pair(ParallelType::OpenMP, &CGwmMGWR::mBandwidthSizeCriterionAllCVOmp),
        #endif
            std::make_pair(ParallelType::CUDA, &CGwmMGWR::mBandwidthSizeCriterionAllCVSerial)
        }),
        std::make_pair<BandwidthSelectionCriterionType, unordered_map<ParallelType, BandwidthSizeCriterionFunction> >(BandwidthSelectionCriterionType::AIC, {
            std::make_pair(ParallelType::SerialOnly, &CGwmMGWR::mBandwidthSizeCriterionAllAICSerial),
        #ifdef ENABLE_OPENMP
            std::make_pair(ParallelType::OpenMP, &CGwmMGWR::mBandwidthSizeCriterionAllAICOmp),
        #endif
            std::make_pair(ParallelType::CUDA, &CGwmMGWR::mBandwidthSizeCriterionAllAICSerial)
        })
    };
    return mapper[type][mParallelType];
}

CGwmMGWR::BandwidthSizeCriterionFunction CGwmMGWR::bandwidthSizeCriterionVar(CGwmMGWR::BandwidthSelectionCriterionType type)
{
    unordered_map<BandwidthSelectionCriterionType, unordered_map<ParallelType, BandwidthSizeCriterionFunction> > mapper = {
        std::make_pair<BandwidthSelectionCriterionType, unordered_map<ParallelType, BandwidthSizeCriterionFunction> >(BandwidthSelectionCriterionType::CV, {
            std::make_pair(ParallelType::SerialOnly, &CGwmMGWR::mBandwidthSizeCriterionVarCVSerial),
        #ifdef ENABLE_OPENMP
            std::make_pair(ParallelType::OpenMP, &CGwmMGWR::mBandwidthSizeCriterionVarCVOmp),
        #endif
            std::make_pair(ParallelType::CUDA, &CGwmMGWR::mBandwidthSizeCriterionVarCVSerial)
        }),
        std::make_pair<BandwidthSelectionCriterionType, unordered_map<ParallelType, BandwidthSizeCriterionFunction> >(BandwidthSelectionCriterionType::AIC, {
            std::make_pair(ParallelType::SerialOnly, &CGwmMGWR::mBandwidthSizeCriterionVarAICSerial),
        #ifdef ENABLE_OPENMP
            std::make_pair(ParallelType::OpenMP, &CGwmMGWR::mBandwidthSizeCriterionVarAICOmp),
        #endif
            std::make_pair(ParallelType::CUDA, &CGwmMGWR::mBandwidthSizeCriterionVarAICSerial)
        })
    };
    return mapper[type][mParallelType];
}

void CGwmMGWR::setParallelType(const ParallelType &type)
{
    if (parallelAbility() & type)
    {
        mParallelType = type;
        switch (type) {
        case ParallelType::SerialOnly:
            mRegressionAll = &CGwmMGWR::regressionAllSerial;
            mRegressionVar = &CGwmMGWR::regressionVarSerial;
            break;
#ifdef ENABLE_OPENMP
        case ParallelType::OpenMP:
            mRegressionAll = &CGwmMGWR::regressionAllOmp;
            mRegressionVar = &CGwmMGWR::regressionVarOmp;
            break;
#endif
//        case IParallelalbe::ParallelType::CUDA:
//            mRegressionAll = &CGwmMGWR::regressionAllOmp;
//            mRegressionVar = &CGwmMGWR::regressionVarOmp;
//            break;
        default:
            break;
        }
    }
}

void CGwmMGWR::setSpatialWeights(const vector<CGwmSpatialWeight> &spatialWeights)
{
    CGwmSpatialMultiscaleAlgorithm::setSpatialWeights(spatialWeights);
    if (spatialWeights.size() > 0)
    {
        mInitSpatialWeight = spatialWeights[0];
    }
}
