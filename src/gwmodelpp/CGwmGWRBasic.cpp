#include "CGwmGWRBasic.h"

GwmRegressionDiagnostic CGwmGWRBasic::CalcDiagnostic(const mat& x, const vec& y, const mat& betas, const vec& shat)
{
    vec r = y - sum(betas % x, 1);
    double rss = sum(r % r);
    double n = x.n_rows;
    double AIC = n * log(rss / n) + n * log(2 * datum::pi) + n + shat(0);
    double AICc = n * log(rss / n) + n * log(2 * datum::pi) + n * ((n + shat(0)) / (n - 2 - shat(0)));
    double edf = n - 2 * shat(0) + shat(1);
    double enp = 2 * shat(0) - shat(1);
    double yss = sum((y - mean(y)) % (y - mean(y)));
    double r2 = 1 - rss / yss;
    double r2_adj = 1 - (1 - r2) * (n - 1) / (edf - 1);
    return { rss, AIC, AICc, enp, edf, r2, r2_adj };
}

CGwmGWRBasic::CGwmGWRBasic()
{
    
}

CGwmGWRBasic::~CGwmGWRBasic()
{
    
}

void CGwmGWRBasic::run()
{
    createRegressionDistanceParameter();
    _ASSERT(mRegressionDistanceParameter != nullptr);

    setXY(mX, mY, mSourceLayer, mDepVar, mIndepVars);

    if (mHasHatMatrix)
    {
        mat betasSE, S;
        vec shat, qdiag;
        mBetas = regressionHatmatrix(mX, mY, betasSE, shat, qdiag, S);
        uword nDp = mSourceLayer->featureCount();
        mDiagnostic = CalcDiagnostic(mX, mY, mBetas, shat);
        double trS = shat(0), trStS = shat(1);
        double sigmaHat = mDiagnostic.RSS / (nDp - 2 * trS + trStS);
        betasSE = sqrt(sigmaHat * betasSE);
        vec yhat = Fitted(mX, mBetas);
        vec res = mY - yhat;
        vec stu_res = res / sqrt(sigmaHat * qdiag);
        mat betasTV = mBetas / betasSE;
        vec dybar2 = (mY - mean(mY)) % (mY - mean(mY));
        vec dyhat2 = (mY - yhat) % (mY - yhat);
        vec localR2 = vec(nDp, fill::zeros);
        for (uword i = 0; i < nDp; i++)
        {
            vec w = mSpatialWeight.weightVector(mRegressionDistanceParameter);
            double tss = sum(dybar2 % w);
            double rss = sum(dyhat2 % w);
            localR2(i) = (tss - rss) / tss;
        }
        createResultLayer({
            make_tuple(string("%1"), mBetas, NameFormat::VarName),
            make_tuple(string("y"), mY, NameFormat::Fixed),
            make_tuple(string("yhat"), yhat, NameFormat::Fixed),
            make_tuple(string("residual"), res, NameFormat::Fixed),
            make_tuple(string("Stud_residual"), stu_res, NameFormat::Fixed),
            make_tuple(string("SE"), betasSE, NameFormat::PrefixVarName),
            make_tuple(string("TV"), betasTV, NameFormat::PrefixVarName),
            make_tuple(string("localR2"), localR2, NameFormat::Fixed)
        });
    }
    else
    {
        mBetas = regression(mX, mY);
        createResultLayer({
            make_tuple(string("%1"), mBetas, NameFormat::VarName)
        });
    }
}

void CGwmGWRBasic::createRegressionDistanceParameter()
{
    if (mSpatialWeight.distance()->type() == CGwmDistance::DistanceType::CRSDistance || 
        mSpatialWeight.distance()->type() == CGwmDistance::DistanceType::MinkwoskiDistance)
    {
        mRegressionDistanceParameter = new CRSDistanceParameter(mSourceLayer->data(), mSourceLayer->data());
    }
}

void CGwmGWRBasic::createPredictionDistanceParameter()
{
    if (mSpatialWeight.distance()->type() == CGwmDistance::DistanceType::CRSDistance || 
        mSpatialWeight.distance()->type() == CGwmDistance::DistanceType::MinkwoskiDistance)
    {
        mRegressionDistanceParameter = new CRSDistanceParameter(mPredictLayer->data(), mSourceLayer->data());
    }
}

mat CGwmGWRBasic::regressionSerial(const mat& x, const vec& y)
{
    uword nRp = mPredictLayer->featureCount(), nVar = mIndepVars.size() + 1;
    mat betas(nVar, nRp, fill::zeros);
    for (uword i = 0; i < nRp; i++)
    {
        vec w = mSpatialWeight.weightVector(mPredictionDistanceParameter);
        mat xtw = trans(x.each_col() % w);
        mat xtwx = xtw * x;
        mat xtwy = xtw * y;
        try
        {
            mat xtwx_inv = inv_sympd(xtwx);
            betas.col(i) = xtwx_inv * xtwy;
        }
        catch (exception e)
        {
            throw e;
        }
    }
    return betas.t();
}

mat CGwmGWRBasic::regressionHatmatrixSerial(const mat& x, const vec& y, mat& betasSE, vec& shat, vec& qDiag, mat& S)
{
    uword nDp = mSourceLayer->featureCount(), nVar = mIndepVars.size() + 1;
    mat betas(nVar, nDp, fill::zeros);
    betasSE = mat(nVar, nDp, fill::zeros);
    shat = vec(2, fill::zeros);
    qDiag = vec(nDp, fill::zeros);
    S = mat(isStoreS() ? nDp : 1, nDp, fill::zeros);
    for (uword i = 0; i < nDp; i++)
    {
        vec w = mSpatialWeight.weightVector(mRegressionDistanceParameter);
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
            shat(0) += si(0, i);
            shat(1) += det(si * si.t());
            vec p = - si.t();
            p(i) += 1.0;
            qDiag += p % p;
            S.row(isStoreS() ? i : 0) = si;
        }
        catch (std::exception e)
        {
            throw e;
        }
    }
    betasSE = betasSE.t();
    return betas.t();
}

void CGwmGWRBasic::createResultLayer(initializer_list<ResultLayerDataItem> items)
{
    mat layerPoints = hasPredictLayer() ? mPredictLayer->points() : mSourceLayer->points();
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
    
    CGwmSimpleLayer* resultLayer = new CGwmSimpleLayer(layerPoints, layerData, layerFields);
}