#include "CGwmGWPCA.h"

#include <assert.h>

CGwmGWPCA::CGwmGWPCA()
{

}

CGwmGWPCA::~CGwmGWPCA()
{
    delete mDistanceParameter;
}

void CGwmGWPCA::run()
{
    createDistanceParameter();
    setX(mX, mSourceLayer, mVariables);
    mLocalPV = pca(mX, mLoadings, mSDev);
    uvec iWinner = index_max(mLoadings.slice(0), 1);
    for (size_t i = 0; i < mSourceLayer->featureCount(); i++)
    {
        mWinner.push_back(mVariables.at(iWinner(i)).name);
    }
    vector<ResultLayerDataItem> resultLayerData = {
        make_tuple(string("PV"), mLocalPV, NameFormat::PrefixCompName),
        make_tuple(string("local_CP"), sum(mLocalPV, 1), NameFormat::Fixed)
    };
    createResultLayer(resultLayerData);
}

void CGwmGWPCA::setX(mat& x, const CGwmSimpleLayer* layer, const vector<GwmVariable>& variables)
{
    uword nRp = mSourceLayer->featureCount(), nVar = variables.size();
    arma::uvec varIndeces(nVar);
    for (size_t i = 0; i < nVar; i++)
    {
        assert(variables[i].index < layer->data().n_cols);
        varIndeces(i) = variables[i].index;
    }
    x = layer->data().cols(varIndeces);
}

mat CGwmGWPCA::solveSerial(const mat& x, cube& loadings, mat& sdev)
{
    uword nDp = mSourceLayer->featureCount(), nVar = mVariables.size();
    mat d_all(nVar, nDp, arma::fill::zeros);
    vec w0;
    loadings = cube(nDp, nVar, mK, arma::fill::zeros);
    for (uword i = 0; i < nDp; i++)
    {
        vec w = mSpatialWeight.weightVector(mDistanceParameter, i);
        mat V;
        vec d;
        wpca(x, w, V, d);
        w0 = w;
        d_all.col(i) = d;
        for (int j = 0; j < mK; j++)
        {
            loadings.slice(j).row(i) = arma::trans(V.col(j));
        }
    }
    d_all = trans(d_all);
    mat variance = (d_all / sqrt(sum(w0))) % (d_all / sqrt(sum(w0)));
    sdev = sqrt(variance);
    mat pv = variance.cols(0, mK - 1).each_col() % (1.0 / sum(variance, 1)) * 100.0;
    return pv;
}

void CGwmGWPCA::wpca(const mat& x, const vec& w, mat& V, vec & d)
{
    mat xw = x.each_col() % w, U;
    mat centerized = (x.each_row() - sum(xw) / sum(w)).each_col() % sqrt(w);
    svd(U, d, V, centerized);
}

void CGwmGWPCA::createResultLayer(vector<ResultLayerDataItem> items)
{
    mat layerPoints = mSourceLayer->points();
    uword layerFeatureCount = mSourceLayer->featureCount();
    mat layerData(layerFeatureCount, 0);
    vector<string> layerFields;
    for (auto &&i : items)
    {
        string name = get<0>(i);
        mat column = get<1>(i);
        NameFormat nf = get<2>(i);
        layerData = join_rows(layerData, column);
        if (nf == NameFormat::PrefixCompName)
        {
            for (size_t k = 0; k < column.n_cols; k++)
            {
                string variableName = string("Comp.") + to_string(k + 1);
                string fieldName = (nf == NameFormat::PrefixCompName) ? (variableName + "_" + name) : (name + "_" + variableName);
                layerFields.push_back(fieldName);
            }
        }
        else
        {
            layerFields.push_back(name);
        }
    }
    mResultLayer = new CGwmSimpleLayer(layerPoints, layerData, layerFields);
}

bool CGwmGWPCA::isValid()
{
    if (CGwmSpatialAlgorithm::isValid())
    {
        if (mK > 0)
        {
            return true;
        }
        else return false;
    }
    else return false;
}
