#include "CGwmGeoWeightedRegression.h"

CGwmGeoWeightedRegression::CGwmGeoWeightedRegression()
{
    
}

CGwmGeoWeightedRegression::~CGwmGeoWeightedRegression()
{

}

bool CGwmGeoWeightedRegression::isValid()
{
    if (CGwmSpatialMonoscaleAlgorithm::isValid())
    {
        if (mIndepVars.size() < 1)
            return false;

        return true;
    }
    else return false;
}

void CGwmGeoWeightedRegression::initXY(mat& x, mat& y, const GwmVariable& depVar, const vector<GwmVariable>& indepVars)
{
    uword nDp = mSourceLayer->featureCount(), nVar = indepVars.size() + 1;
    arma::uvec indepVarIndeces(indepVars.size());
    for (size_t i = 0; i < indepVars.size(); i++)
    {
        _ASSERT(indepVars[i].index < x.n_cols);
        indepVarIndeces(i) = indepVars[i].index;
    }
    x = join_rows(mat(nDp, 1, arma::fill::ones), mSourceLayer->data().cols(indepVarIndeces));
    y = mSourceLayer->data().col(depVar.index);
}