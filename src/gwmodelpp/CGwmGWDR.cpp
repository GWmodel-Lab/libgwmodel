#include "CGwmGWDR.h"
#include <assert.h>


GwmRegressionDiagnostic CGwmGWDR::CalcDiagnostic(const mat& x, const vec& y, const mat& betas, const vec& shat)
{
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
}

void CGwmGWDR::run()
{
    // Set data matrices.
    setXY(mX, mY, mSourceLayer, mDepVar, mIndepVars);

    // Set coordinates matrices.
    uword nDims = mSourceLayer->points().n_cols;
    for (size_t m = 0; m < nDims; m++)
    {
        DistanceParameter* oneDimDP = new OneDimDistanceParameter(mSourceLayer->points().col(m), mSourceLayer->points().col(m));
        mDistParameters.push_back(oneDimDP);
    }
    
    // Has hatmatrix
    uword nDp = mSourceLayer->featureCount(), nDim = mSourceLayer->points().n_cols;
    if (mHasHatMatrix)
    {
        mat betasSE, S;
        vec shat, qdiag;
        mBetas = regressionHatmatrix(mX, mY, betasSE, shat, qdiag, S);
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
        vec localR2 = vec(nDp, arma::fill::zeros);
        for (uword i = 0; i < nDp; i++)
        {
            vec w(nDp, arma::fill::ones);
            for (size_t m = 0; m < nDim; m++)
            {
                w = w % mSpatialWeights[m].weightVector(mDistParameters[m], i);
            }
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
    
}

void CGwmGWDR::setXY(mat& x, mat& y, const CGwmSimpleLayer* layer, const GwmVariable& depVar, const vector<GwmVariable>& indepVars)
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

mat CGwmGWDR::regressionSerial(const mat& x, const vec& y)
{
    uword nDp = mSourceLayer->featureCount(), nVar = mIndepVars.size() + 1, nDim = mSourceLayer->points().n_cols;
    mat betas(nVar, nDp, arma::fill::zeros);
    for (size_t i = 0; i < nDp; i++)
    {
        vec w(nDp, arma::fill::ones);
        for (size_t m = 0; m < nDim; m++)
        {
            vec w_m = mSpatialWeights[m].weightVector(mDistParameters[m], i);
            w = w % w_m;
        }
        mat ws(1, nVar, arma::fill::ones);
        mat xtw = (x %(w * ws)).t();
        mat xtwx = xtw * x;
        mat xtwy = x.t() * (w % y);
        try
        {
            mat xtwx_inv = xtwx.i();
            betas.col(i) = xtwx_inv * xtwy;
        }
        catch(const std::exception& e)
        {
            std::cerr << e.what() << '\n';
        }
    }
    return betas.t();
}

mat CGwmGWDR::regressionHatmatrixSerial(const mat& x, const vec& y, mat& betasSE, vec& shat, vec& qdiag, mat& S)
{
    uword nDp = mSourceLayer->featureCount(), nVar = mIndepVars.size() + 1, nDim = mSourceLayer->points().n_cols;
    mat betas(nVar, nDp, arma::fill::zeros);
    betasSE = mat(nVar, nDp, arma::fill::zeros);
    qdiag = vec(nDp, arma::fill::zeros);
    S = mat(isStoreS() ? nDp : 1, nDp, arma::fill::zeros);
    mat rowsumSE(nDp, 1, arma::fill::ones);
    vec s_hat1(nDp, arma::fill::zeros), s_hat2(nDp, arma::fill::zeros);
    for (size_t i = 0; i < nDp; i++)
    {
        vec w(nDp, arma::fill::ones);
        for (size_t m = 0; m < nDim; m++)
        {
            vec w_m = mSpatialWeights[m].weightVector(mDistParameters[m], i);
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

void CGwmGWDR::createResultLayer(initializer_list<ResultLayerDataItem> items)
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
}
