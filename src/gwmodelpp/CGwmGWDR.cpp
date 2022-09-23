#include "CGwmGWDR.h"
#include <assert.h>
#include <exception>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_multimin.h>
#include <gsl/gsl_errno.h>

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
    uword nDp = mSourceLayer->featureCount(), nDims = mSourceLayer->points().n_cols;

    // Set coordinates matrices.
    for (size_t m = 0; m < nDims; m++)
    {
        DistanceParameter* oneDimDP = mSpatialWeights[m].distance()->makeParameter({
            vec(mSourceLayer->points().col(m)),
            vec(mSourceLayer->points().col(m))
        });
        mDistParameters.push_back(oneDimDP);
    }

    // Select Independent Variable
    if (mEnableIndepVarSelect)
    {
        CGwmVariableForwardSelector selector(mIndepVars, mIndepVarSelectThreshold);
        vector<GwmVariable> selectedIndepVars = selector.optimize(this);
        if (selectedIndepVars.size() > 0)
        {
            mIndepVars = selectedIndepVars;
            mIndepVarCriterionList = selector.indepVarsCriterion();
        }
    }
    

    // Set data matrices.
    setXY(mX, mY, mSourceLayer, mDepVar, mIndepVars);

    if (mEnableBandwidthOptimize)
    {
        for (size_t m = 0; m < mSpatialWeights.size(); m++)
        {
            const CGwmSpatialWeight& sw = mSpatialWeights[m];
            CGwmBandwidthWeight* bw = sw.weight<CGwmBandwidthWeight>();
            // Set Initial value
            if (bw->bandwidth() == 0.0)
            {
                double upper = bw->adaptive() ? nDp : sw.distance()->maxDistance(nDp, mDistParameters[m]);
                bw->setBandwidth(upper * 0.618);
            }
        }
        vector<CGwmBandwidthWeight*> bws;
        for (auto&& iter : mSpatialWeights)
        {
            bws.push_back(iter.weight<CGwmBandwidthWeight>());
        }
        CGwmGWDRBandwidthOptimizer optimizer(bws);
        int status = optimizer.optimize(this, mSourceLayer->featureCount(), 100000, 1e-8);
        if (status)
        {
            throw runtime_error("[CGwmGWDR::run] Bandwidth optimization invoke failed.");
        }
    }
    
    // Has hatmatrix
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
            for (size_t m = 0; m < nDims; m++)
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

double CGwmGWDR::bandwidthCriterionCVSerial(const vector<CGwmBandwidthWeight*>& bandwidths)
{
    uword nDp = mSourceLayer->featureCount(), nVar = mIndepVars.size() + 1, nDim = mSourceLayer->points().n_cols;
    double cv = 0.0;
    bool flag = true;
    for (size_t i = 0; i < nDp; i++)
    {
        if (flag)
        {
            vec w(nDp, arma::fill::ones);
            for (size_t m = 0; m < nDim; m++)
            {
                vec d_m = mSpatialWeights[m].distance()->distance(mDistParameters[m], i);
                vec w_m = bandwidths[m]->weight(d_m);
                w = w % w_m;
            }
            w(i) = 0.0;
            mat ws(1, nVar, arma::fill::ones);
            mat xtw = (mX %(w * ws)).t();
            mat xtwx = xtw * mX;
            mat xtwy = mX.t() * (w % mY);
            try
            {
                mat xtwx_inv = xtwx.i();
                vec beta = xtwx_inv * xtwy;
                double yhat = as_scalar(mX.row(i) * beta);
                cv += yhat - mY(i);
            }
            catch(const std::exception& e)
            {
                std::cerr << e.what() << '\n';
                flag = false;
            }
        }
    }
    return flag ? abs(cv) : DBL_MAX;
}

double CGwmGWDR::bandwidthCriterionAICSerial(const vector<CGwmBandwidthWeight*>& bandwidths)
{
    uword nDp = mSourceLayer->featureCount(), nVar = mIndepVars.size() + 1, nDim = mSourceLayer->points().n_cols;
    mat betas(nVar, nDp, fill::zeros);
    double trS = 0.0;
    bool flag = true;
    for (size_t i = 0; i < nDp; i++)
    {
        if (flag)
        {
            vec w(nDp, arma::fill::ones);
            for (size_t m = 0; m < nDim; m++)
            {
                vec d_m = mSpatialWeights[m].distance()->distance(mDistParameters[m], i);
                vec w_m = bandwidths[m]->weight(d_m);
                w = w % w_m;
            }
            mat ws(1, nVar, arma::fill::ones);
            mat xtw = (mX %(w * ws)).t();
            mat xtwx = xtw * mX;
            mat xtwy = mX.t() * (w % mY);
            try
            {
                mat xtwx_inv = xtwx.i();
                betas.col(i) = xtwx_inv * xtwy;
                mat ci = xtwx_inv * xtw;
                mat si = mX.row(i) * ci;
                trS += si(0, i);
            }
            catch(const std::exception& e)
            {
                std::cerr << e.what() << '\n';
                flag = false;
            }
        }
    }
    if (!flag) return DBL_MAX;
    double value = CGwmGWDR::AICc(mX, mY, betas.t(), { trS, 0.0 });
    return isfinite(value) ? value : DBL_MAX;
}

double CGwmGWDR::indepVarCriterionSerial(const vector<GwmVariable>& indepVars)
{
    mat x;
    vec y;
    setXY(x, y, mSourceLayer, mDepVar, indepVars);
    uword nDp = x.n_rows, nVar = x.n_cols;
    mat betas(nVar, nDp, fill::zeros);
    double trS = 0.0;
    bool isGlobal = false, success = true;
    if (mEnableBandwidthOptimize) isGlobal = true;
    else
    {
        for (auto &&sw : mSpatialWeights)
        {
            if (sw.weight<CGwmBandwidthWeight>()->bandwidth() == 0.0) isGlobal = true;
        }
    }
    if (isGlobal)
    {
        mat xtwx = x.t() * x;
        vec xtwy = x.t() * y;
        try
        {
            mat xtwx_inv = inv_sympd(xtwx);
            vec beta = xtwx_inv * xtwy;
            betas = betas.each_col() + beta;
            mat ci = xtwx_inv * x.t();
            for (uword i = 0; i < nDp; i++)
            {
                mat si = x.row(i) * ci;
                trS += si(0, i);
            }
        }
        catch(const std::exception& e)
        {
            std::cerr << e.what() << '\n';
            success = false;
        }
        
    }
    else
    {
        int nDim = mSpatialWeights.size();
        for (uword i = 0; i < nDp; i++)
        {
            if (success)
            {
                vec w(nDp, arma::fill::ones);
                for (size_t m = 0; m < nDim; m++)
                {
                    vec w_m = mSpatialWeights[m].weightVector(mDistParameters[m], i);
                    w = w % w_m;
                }
                mat xtw = (x.each_col() % w).t();
                mat xtwx = xtw * x;
                mat xtwy = xtw * y;
                try
                {
                    mat xtwx_inv = inv_sympd(xtwx);
                    betas.col(i) = xtwx_inv * xtwy;
                    mat ci = xtwx_inv * xtw;
                    mat si = x.row(i) * ci;
                    trS += si(0, i);
                }
                catch (std::exception& e)
                {
                    std::cerr << e.what() << '\n';
                    success = false;
                }
            }
        }
    }
    double value = success ? CGwmGWDR::AICc(x, y, betas.t(), { trS, 0.0 }) : DBL_MAX;
    // string msg = "Model: " + mDepVar.name + " ~ ";
    // for (size_t i = 0; i < indepVars.size() - 1; i++)
    // {
    //     msg += indepVars[i].name + " + ";
    // }
    // msg += indepVars.back().name;
    // msg += " (AICc Value: " + to_string(value) + ")";
    return isfinite(value) ? value : DBL_MAX;
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

void CGwmGWDR::setBandwidthCriterionType(const BandwidthCriterionType& type)
{
    mBandwidthCriterionType = type;
    switch (mBandwidthCriterionType)
    {
    case BandwidthCriterionType::AIC:
        mBandwidthCriterionFunction = &CGwmGWDR::bandwidthCriterionAICSerial;
        break;
    default:
        mBandwidthCriterionFunction = &CGwmGWDR::bandwidthCriterionCVSerial;
        break;
    }
}

double CGwmGWDRBandwidthOptimizer::criterion_function(const gsl_vector* bws, void* params)
{
    Parameter* p = static_cast<Parameter*>(params);
    CGwmGWDR* instance = p->instance;
    const vector<CGwmBandwidthWeight*>& bandwidths = *(p->bandwidths);
    double nFeature = double(p->featureCount);
    for (size_t m = 0; m < bandwidths.size(); m++)
    {
        double pbw = abs(gsl_vector_get(bws, m));
        bandwidths[m]->setBandwidth(pbw * nFeature);
    }
    return instance->bandwidthCriterion(bandwidths);
}

const int CGwmGWDRBandwidthOptimizer::optimize(CGwmGWDR* instance, uword featureCount, size_t maxIter, double eps)
{
    size_t nDim = mBandwidths.size();
    gsl_multimin_fminimizer* minimizer = gsl_multimin_fminimizer_alloc(gsl_multimin_fminimizer_nmsimplex, nDim);
    gsl_vector* target = gsl_vector_alloc(nDim);
    gsl_vector* step = gsl_vector_alloc(nDim);
    for (size_t m = 0; m < nDim; m++)
    {
        double target_value = mBandwidths[m]->adaptive() ? mBandwidths[m]->bandwidth() / double(featureCount) : mBandwidths[m]->bandwidth();
        gsl_vector_set(target, m, target_value);
        gsl_vector_set(step, m, 0.1);
    }
    Parameter params = { instance, &mBandwidths, featureCount };
    gsl_multimin_function function = { criterion_function, nDim, &params };
    double criterion = DBL_MAX;
    int status = gsl_multimin_fminimizer_set(minimizer, &function, target, step);
    if (status == GSL_SUCCESS)
    {
        int iter = 0;
        double size = DBL_MAX, size0 = DBL_MAX;
        do
        {
            size0 = size;
            iter++;
            status = gsl_multimin_fminimizer_iterate(minimizer);
            if (status)
                break;
            size = gsl_multimin_fminimizer_size(minimizer);
            status = gsl_multimin_test_size(size, eps);
            #ifdef _DEBUG
            for (size_t m = 0; m < nDim; m++)
            {
                cout << gsl_vector_get(minimizer->x, m) << ",";
            }
            cout << minimizer->fval << ",";
            cout << size << "\n";
            #endif
        } 
        while (status == GSL_CONTINUE && iter < maxIter);
        #ifdef _DEBUG
        for (size_t m = 0; m < nDim; m++)
        {
            cout << gsl_vector_get(minimizer->x, m) << ",";
        }
        cout << minimizer->fval << ",";
        cout << size << "\n";
        #endif
        for (size_t m = 0; m < nDim; m++)
        {
            double pbw = abs(gsl_vector_get(minimizer->x, m));
            // pbw = (pbw > 1.0 ? 1.0 : pbw);
            mBandwidths[m]->setBandwidth(round(pbw * featureCount));
        }
    }
    gsl_multimin_fminimizer_free(minimizer);
    gsl_vector_free(target);
    gsl_vector_free(step);
    return status;
}
