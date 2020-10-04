#include "CGwmGWSS.h"

#ifdef ENABLE_OPENMP
#include <omp.h>
#endif

vec CGwmGWSS::del(vec x, int rowcount){
    vec res;
    if(rowcount == 0)
        res = x.rows(rowcount+1,x.n_rows-1);
    else if(rowcount == x.n_rows-1)
        res = x.rows(0,x.n_rows-2);
    else
        res = join_cols(x.rows(0,rowcount - 1),x.rows(rowcount+1,x.n_rows-1));
    return res;
}

vec CGwmGWSS::findq(const mat &x, const vec &w)
{
    uword lw = w.n_rows;
    uword lp = 3;
    vec q = vec(lp,fill::zeros);
    vec xo = sort(x);
    vec wo = w(sort_index(x));
    vec Cum = cumsum(wo);
    uword cond = lw - 1;
    for(uword j = 0; j < lp ; j++){
        double k = 0.25 * (j + 1);
        for(uword i = 0; i < lw; i++){
            if(Cum(i) > k){
                cond = i - 1;
                break;
            }
        }
        if(cond < 0)
        {
            cond = 0;
        }
        q.row(j) = xo[cond];
        cond = lw - 1;
    }
    return q;
}

CGwmGWSS::CGwmGWSS()
{

}

CGwmGWSS::~CGwmGWSS()
{

}

bool CGwmGWSS::isValid()
{
    if (CGwmSpatialMonoscaleAlgorithm::isValid())
    {
        if (mVariables.size() < 1)
            return false;

        return true;
    }
    else return false;
}

void CGwmGWSS::run()
{
    createDistanceParameter();
    setXY(mX, mSourceLayer, mVariables);
    (this->*mSummaryFunction)();
    vector<ResultLayerDataItem> resultLayerData = {
        make_tuple(string("LM"), mLocalMean, NameFormat::PrefixVarName),
        make_tuple(string("LSD"), mStandardDev, NameFormat::PrefixVarName),
        make_tuple(string("LVar"), mLVar, NameFormat::PrefixVarName),
        make_tuple(string("LSke"), mLocalSkewness, NameFormat::PrefixVarName),
        make_tuple(string("LCV"), mLCV, NameFormat::PrefixVarName)
    };
    if (mQuantile)
    {
        resultLayerData.push_back(make_tuple(string("Median"), mLocalMedian, NameFormat::PrefixVarName));
        resultLayerData.push_back(make_tuple(string("IQR"), mIQR, NameFormat::PrefixVarName));
        resultLayerData.push_back(make_tuple(string("QI"), mQI, NameFormat::PrefixVarName));
    }
    if (mVariables.size() > 1)
    {
        resultLayerData.push_back(make_tuple(string("Cov"), mCovmat, NameFormat::SuffixVarNamePair));
        resultLayerData.push_back(make_tuple(string("Corr"), mCorrmat, NameFormat::SuffixVarNamePair));
        resultLayerData.push_back(make_tuple(string("Spearman_rho"), mSCorrmat, NameFormat::SuffixVarNamePair));
    }
    createResultLayer(resultLayerData);
}

void CGwmGWSS::setXY(mat& x, const CGwmSimpleLayer* layer, const vector<GwmVariable>& variables)
{
    uword nRp = mSourceLayer->featureCount(), nVar = variables.size();
    arma::uvec varIndeces(nVar);
    for (size_t i = 0; i < nVar; i++)
    {
        _ASSERT(variables[i].index < layer->data().n_cols);
        varIndeces(i) = variables[i].index;
    }
    x = layer->data().cols(varIndeces);
    mLocalMean = mat(nRp, nVar, fill::zeros);
    mStandardDev = mat(nRp, nVar, fill::zeros);
    mLocalSkewness = mat(nRp, nVar, fill::zeros);
    mLCV = mat(nRp, nVar, fill::zeros);
    mLVar = mat(nRp, nVar, fill::zeros);
    if (mQuantile)
    {
        mLocalMedian = mat(nRp, nVar, fill::zeros);
        mIQR = mat(nRp, nVar, fill::zeros);
        mQI = mat(nRp, nVar, fill::zeros);
    }
    if (nVar > 1)
    {
        uword nCol = mIsCorrWithFirstOnly ? (nVar - 1) : (nVar + 1) * nVar / 2 - nVar;
        mCovmat = mat(nRp, nCol, fill::zeros);
        mCorrmat = mat(nRp, nCol, fill::zeros);
        mSCorrmat = mat(nRp, nCol, fill::zeros);
    }
}

void CGwmGWSS::createDistanceParameter()
{
    if (mSpatialWeight.distance()->type() == CGwmDistance::DistanceType::CRSDistance || 
        mSpatialWeight.distance()->type() == CGwmDistance::DistanceType::MinkwoskiDistance)
    {
        mDistanceParameter = new CRSDistanceParameter(mSourceLayer->points(), mSourceLayer->points());
    }
}

void CGwmGWSS::summarySerial()
{
    mat rankX = mX;
    rankX.each_col([&](vec& x) { x = rank(x); });
    uword nVar = mX.n_cols, nRp = mSourceLayer->featureCount();
    uword corrSize = mIsCorrWithFirstOnly ? 1 : nVar - 1;
    for (uword i = 0; i < nRp; i++)
    {
        vec w = mSpatialWeight.weightVector(mDistanceParameter, i);
        double sumw = sum(w);
        vec Wi = w / sumw;
        mLocalMean.row(i) = trans(Wi) * mX;
        if (mQuantile)
        {
            mat quant = mat(3, nVar);
            for (uword j = 0; j < nVar; j++)
            {
                quant.col(j) = findq(mX.col(j), Wi);
            }
            mLocalMedian.row(i) = quant.row(1);
            mIQR.row(i) = quant.row(2) - quant.row(0);
            mQI.row(i) = (2 * quant.row(1) - quant.row(2) - quant.row(0)) / mIQR.row(i);
        }
        mat centerized = mX.each_row() - mLocalMean.row(i);
        mLVar.row(i) = Wi.t() * (centerized % centerized);
        mStandardDev.row(i) = sqrt(mLVar.row(i));
        mLocalSkewness.row(i) = (Wi.t() * (centerized % centerized % centerized)) / (mLVar.row(i) % mStandardDev.row(i));
        if (nVar >= 2)
        {
            uword tag = 0;
            for (uword j = 0; j < corrSize; j++)
            {
                for (uword k = j + 1; k < nVar; k++)
                {
                    double covjk = covwt(mX.col(j), mX.col(k), Wi);
                    double sumW2 = sum(Wi % Wi);
                    double covjj = mLVar(i, j) / (1.0 - sumW2);
                    double covkk = mLVar(i, k) / (1.0 - sumW2);
                    mCovmat(i, tag) = covjk;
                    mCorrmat(i, tag) = covjk / sqrt(covjj * covkk);
                    mSCorrmat(i, tag) = corwt(rankX.col(j), rankX.col(k), Wi);
                    tag++;
                }
            }
        }
    }
    mLCV = mStandardDev / mLocalMean;
}

#ifdef ENABLE_OPENMP
void CGwmGWSS::summaryOmp()
{
    mat rankX = mX;
    rankX.each_col([&](vec& x) { x = rank(x); });
    uword nVar = mX.n_cols, nRp = mSourceLayer->featureCount();
    uword corrSize = mIsCorrWithFirstOnly ? 1 : nVar - 1;
#pragma omp parallel for num_threads(mOmpThreadNum)
    for (int i = 0; (uword)i < nRp; i++)
    {
        int thread = omp_get_thread_num();
        vec w = mSpatialWeight.weightVector(mDistanceParameter, i);
        double sumw = sum(w);
        vec Wi = w / sumw;
        mLocalMean.row(i) = trans(Wi) * mX;
        if (mQuantile)
        {
            mat quant = mat(3, nVar);
            for (uword j = 0; j < nVar; j++)
            {
                quant.col(j) = findq(mX.col(j), Wi);
            }
            mLocalMedian.row(i) = quant.row(1);
            mIQR.row(i) = quant.row(2) - quant.row(0);
            mQI.row(i) = (2 * quant.row(1) - quant.row(2) - quant.row(0)) / mIQR.row(i);
        }
        mat centerized = mX.each_row() - mLocalMean.row(i);
        mLVar.row(i) = Wi.t() * (centerized % centerized);
        mStandardDev.row(i) = sqrt(mLVar.row(i));
        mLocalSkewness.row(i) = (Wi.t() * (centerized % centerized % centerized)) / (mLVar.row(i) % mStandardDev.row(i));
        if (nVar >= 2)
        {
            uword tag = 0;
            for (uword j = 0; j < corrSize; j++)
            {
                for (uword k = j + 1; k < nVar; k++)
                {
                    double covjk = covwt(mX.col(j), mX.col(k), Wi);
                    double sumW2 = sum(Wi % Wi);
                    double covjj = mLVar(i, j) / (1.0 - sumW2);
                    double covkk = mLVar(i, k) / (1.0 - sumW2);
                    mCovmat(i, tag) = covjk;
                    mCorrmat(i, tag) = covjk / sqrt(covjj * covkk);
                    mSCorrmat(i, tag) = corwt(rankX.col(j), rankX.col(k), Wi);
                    tag++;
                }
            }
        }
    }
    mLCV = mStandardDev / mLocalMean;
    
}
#endif

void CGwmGWSS::createResultLayer(vector<ResultLayerDataItem> items)
{
    mat layerPoints = mSourceLayer->points();
    uword layerFeatureCount = mSourceLayer->featureCount();
    mat layerData(layerFeatureCount, 0);
    vector<string> layerFields;
    for (auto &&i : items)
    {
        NameFormat nf = get<2>(i);
        mat column = get<1>(i);
        string name = get<0>(i);
        layerData = join_rows(layerData, column);
        if (nf == NameFormat::PrefixVarName || nf == NameFormat::SuffixVarName)
        {
            for (size_t k = 0; k < column.n_cols; k++)
            {
                string variableName = mVariables[k].name;
                string fieldName = (nf == NameFormat::PrefixVarName) ? (variableName + "_" + name) : (name + "_" + variableName);
                layerFields.push_back(fieldName);
            }
        }
        else if (nf == NameFormat::PrefixVarNamePair || nf == NameFormat::SuffixVarNamePair)
        {
            uword nVar = mVariables.size();
            uword corrVarSize = mIsCorrWithFirstOnly ? 1 : (nVar - 1);
            for (size_t i = 0; i < corrVarSize; i++)
            {
                string var1name = mVariables[i].name;
                for (size_t j = i + 1; j < nVar; j++)
                {
                    string var2name = mVariables[j].name;
                    string varNamePair = var1name + "." + var2name;
                    string fieldName = (nf == NameFormat::PrefixVarNamePair) ? (varNamePair + "_" + name) : (name + "_" + varNamePair);
                    layerFields.push_back(fieldName);
                }
            }
        }
    }
    
    mResultLayer = new CGwmSimpleLayer(layerPoints, layerData, layerFields);
}

void CGwmGWSS::setParallelType(const ParallelType &type)
{
    if (type & parallelAbility())
    {
        mParallelType = type;
        switch (type) {
        case ParallelType::SerialOnly:
            mSummaryFunction = &CGwmGWSS::summarySerial;
            break;
        case ParallelType::OpenMP:
            mSummaryFunction = &CGwmGWSS::summaryOmp;
            break;
        default:
            mSummaryFunction = &CGwmGWSS::summarySerial;
            break;
        }
    }
}