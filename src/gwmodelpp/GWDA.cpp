#include "GWDA.h"
#include <assert.h>
#include <vector>
#include <string>

#ifdef ENABLE_OPENMP
#include <omp.h>
#endif

using namespace arma;
using namespace gwm;
using namespace std;

// template<class T>
bool GWDA::isValid()
{
    if (SpatialAlgorithm::isValid())
    {
        if (!(mX.n_cols > 0))
            return false;

        return true;
    }
    else
        return false;
}

// template<class T>
void GWDA::run()
{
    createDistanceParameter();
    uword nRp = mCoords.n_rows;
    if(mX.n_cols<2){
        throw std::runtime_error("Two or more variables should be specfied for analysis");
    }
    // vec lev = levels(mY);
    mHasPredict = !(mprX.is_empty() && mprY.empty());
    if (mprX.is_empty())
    {
        mprX = mX;
    }
    if (mprY.empty())
    {
        mprY = mY;
    }
    (this->*mDiscriminantAnalysisFunction)();
    uword NV = mRes.n_cols;
    vec correct(mY.size(), fill::zeros);
    for (uword i = 0; i < mGroup.size(); i++)
    {
        if (mGroup[i] == mY[i])
        {
            correct[i] = 1;
        }
    }
    uvec correctCount = find(correct == 1);
    mCorrectRate = (double)correctCount.n_rows / nRp;
    mat tmp = mRes.cols(0, NV - 1);
    for (uword i = 0; i < NV - 1; i++)
    {
        vec tempi = tmp.col(i);
        tmp.col(i) = exp(tempi);
    }
    mat probs = tmp.each_col() / sum(tmp, 1); // / sum(tmp, 1);
    vec pmax = max(probs, 1);
    // double pnorm=0;
    vec p = vec(NV - 1, fill::ones) / (NV - 1.0);
    double entMax = shannonEntropy(p);
    vec entropy(nRp, fill::zeros);
    for (uword i = 0; i < nRp; i++)
    {
        vec t = probs.row(i).t();
        entropy(i) = shannonEntropy(t) / entMax;
    }
    mProbs = probs;
    mPmax = pmax;
    mEntropy = entropy;
}

// template<class T>
double GWDA::shannonEntropy(arma::vec &p)
{
    double entMax = 0;
    if (min(p) < 0 || sum(p) <= 0)
    {
        return entMax;
    }
    vec pnorm = p(find(p > 0)) / sum(p);
    entMax = -sum(log2(pnorm) % pnorm);
    return entMax;
}

// template<class T>
void GWDA::discriminantAnalysisSerial()
{
    uword nRp = mCoords.n_rows;
    // uword nVar = mX.n_cols;//nPr = mprX.n_rows;
    vector<string> lev = levels(mY);
    mat wt(nRp, nRp, fill::zeros);
    for (uword i = 0; i < nRp; i++)
    {
        // vec w=mSpatialWeight.weightVector(i);
        wt.col(i) = mSpatialWeight.weightVector(i);
    }
    if (!mHasPredict) wt.diag().zeros();
    if (mIsWqda)
    {
        mRes = wqda(mX, mY, wt, mprX, mHascov, mHasmean, mHasprior);
    }
    else
    {
        mRes = wlda(mX, mY, wt, mprX, mHascov, mHasmean, mHasprior);
    }
}

#ifdef ENABLE_OPENMP
// template<class T>
void GWDA::discriminantAnalysisOmp()
{
    uword nRp = mCoords.n_rows;
    // uword nVar = mX.n_cols;//nPr = mprX.n_rows;
    vector<string> lev = levels(mY);
    mat wt(nRp, nRp, fill::zeros);
#pragma omp parallel for num_threads(mOmpThreadNum)
    for (uword i = 0; i < nRp; i++)
    {
        wt.col(i) = mSpatialWeight.weightVector(i);
    }
    if (mIsWqda)
    {
        mRes = wqda(mX, mY, wt, mprX, mHascov, mHasmean, mHasprior);
    }
    else
    {
        mRes = wlda(mX, mY, wt, mprX, mHascov, mHasmean, mHasprior);
    }
}
#endif

uvec GWDA::findSameString(std::vector<std::string> &y, std::string s)
{
    uvec flags(y.size());
    transform(y.cbegin(), y.cend(), flags.begin(), [&s](const string& ys)
    {
        return uword(ys == s ? 1 : 0); 
    });
    return find(flags == 1);
}

// template<class T>
vector<mat> GWDA::splitX(arma::mat &x, std::vector<std::string> &y)
{
    vector<string> lev = levels(y);
    uword p = lev.size();
    vector<mat> res;
    for (uword i = 0; i < p; i++)
    {
        res.push_back(x.rows(findSameString(y, lev[i])));
    }
    return res;
}

// template<class T>
vector<string> GWDA::levels(vector<std::string> &y)
{
    uword n = y.size();
    vector<string> lev;
    for (uword i = 0; i < n; i++)
    {
        auto d = std::find(lev.begin(), lev.end(), y[i]);
        if (d == lev.end())
        {
            lev.push_back(y[i]);
        }
    }
    return lev;
}

unordered_map<string, uword> GWDA::ytable(std::vector<std::string> &y)
{
    uword n = y.size();
    unordered_map<string, uword> counts;
    for (uword i = 0; i < n; i++)
    {
        unordered_map<string, uword>::const_iterator d = counts.find(y[i]);
        if (d == counts.end())
        {
            counts.insert(make_pair(y[i], 1));
        }
        else
        {
            counts[y[i]] = (d->second) + 1;
        }
    }
    return counts;
}

// template<class T>
mat GWDA::covwtmat(const arma::mat &x, const arma::vec &wt)
{
    uword n = x.n_cols;
    mat sigma(n, n, fill::zeros);
    double w = sum(wt) / (sum(wt) * sum(wt) - sum(wt % wt));
    vec average(n, fill::zeros);
    for (uword i = 0; i < n; i++)
    {
        average(i) = sum(x.col(i) % wt) / sum(wt);
    }
    for (uword j = 0; j < n; j++)
    {
        for (uword k = 0; k < n; k++)
        {
            sigma(j, k) = w * sum(wt % ((x.col(j) - average(j)) % (x.col(k) - average(k))));
        }
    }
    return sigma;
}

// template<class T>
mat GWDA::wqda(arma::mat &x, std::vector<std::string> &y, arma::mat &wt, arma::mat &xpr, bool hasCOv, bool hasMean, bool hasPrior)
{
    vector<string> lev = levels(y);
    uword m = lev.size();
    // uword nDp = x.n_rows;
    uword nPr = xpr.n_rows;
    mat wtOnes;
    // wtOnes.ones(nDp, nPr);
    vector<mat> xg = splitX(x, y);
    vector<vec> prior;
    mat wti; //(nDp, nPr, fill::zeros);
    vector<mat> localMean;
    vector<cube> sigmagw;
    double sumW;
    for (uword i = 0; i < m; i++)
    {
        mat xi = xg[i];
        /*vector<uword> idx;
        // uword index = 0;
        for (uword j = 0; j < y.size(); j++)
        {
            if (y[j] == lev[i])
            {
                idx.push_back(j);
            }
        }*/
        if (hasMean)
        {
            /*for (uword j = 0; j < idx.size(); j++)
            {
                wti.row(idx[j]) = mSpatialWeight.weightVector(j);
            }*/
            wti = wt.rows(findSameString(y, lev[i]));
        }
        else
        {
            wti = wtOnes.rows(findSameString(y, lev[i]));
        }
        localMean.push_back(wMean(xi, wti));
        if (hasCOv)
        {
            wti = wt.rows(findSameString(y, lev[i]));
        }
        else
        {
            wti = wtOnes.rows(findSameString(y, lev[i]));
        }
        sigmagw.push_back(wVarCov(xi, wti));
        if (hasPrior)
        {
            wti = wt.rows(findSameString(y, lev[i]));
            sumW = accu(wt);
        }
        else
        {
            wti = wtOnes.rows(findSameString(y, lev[i]));
            sumW = accu(wtOnes);
        }
        prior.push_back(wPrior(wti, sumW));
    }
    mat logPf = mat(nPr, m, fill::zeros);
    for (uword i = 0; i < m; i++)
    {
        for (uword j = 0; j < nPr; j++)
        {
            vec xprj = xpr.row(j).t();
            vec meani = localMean[i].row(j).t();
            mat covmatj = sigmagw[i].row(j);
            vec x1 = 0.5 * (xprj - meani).t() * solve(covmatj, (xprj - meani)); // inv(covmatj) * (xprj - meani);
            logPf(j, i) = (m / 2) * log(norm(covmatj)) + x1(0) - log(prior[i].at(j));
        }
    }
    vector<string> groupPr;
    for (uword i = 0; i < nPr; i++)
    {
        uword index = index_min(logPf.row(i));
        groupPr.push_back((lev[index]));
    }
    mGroup = groupPr;
    return logPf;
}

// template<class T>
mat GWDA::wlda(arma::mat &x, std::vector<std::string> &y, arma::mat &wt, arma::mat &xpr, bool hasCOv, bool hasMean, bool hasPrior)
{
    vector<string> lev = levels(y);
    uword m = lev.size();
    uword nDp = x.n_rows;
    uword nPr = xpr.n_rows;
    uword nVar = x.n_cols;
    mat wtOnes;
    wtOnes.ones(nDp, nPr);
    vector<mat> xg = splitX(x, y);
    vector<vec> prior;
    mat wti(nDp, nPr, fill::zeros);
    vector<mat> localMean;
    vector<cube> sigmagw;
    double sumW;
    for (uword i = 0; i < m; i++)
    {
        mat xi = xg[i];
        /*vector<uword> idx;
        // uword index = 0;
        for (uword j = 0; j < y.size(); j++)
        {
            if (y[j] == lev[i])
            {
                idx.push_back(j);
            }
        }*/
        if (hasMean)
        {
            wti = wt.rows(findSameString(y, lev[i]));
        }
        else
        {
            wti = wtOnes.rows(findSameString(y, lev[i]));
        }
        localMean.push_back(wMean(xi, wti));
        if (hasCOv)
        {
            wti = wt.rows(findSameString(y, lev[i]));
        }
        else
        {
            wti = wtOnes.rows(findSameString(y, lev[i]));
        }
        sigmagw.push_back(wVarCov(xi, wti));
        if (hasPrior)
        {
            wti = wt.rows(findSameString(y, lev[i]));
            sumW = accu(wt);
        }
        else
        {
            wti = wtOnes.rows(findSameString(y, lev[i]));
            sumW = accu(wtOnes);
        }
        prior.push_back(wPrior(wti, sumW));
    }
    cube sigma1gw(nPr, nVar, nVar, fill::zeros);
    unordered_map<string, uword> counts = ytable(y);
    for (uword i = 0; i < nPr; i++)
    {
        mat sigmai(nVar, nVar, fill::zeros);
        double yisum = 0;
        for (uword j = 0; j < m; j++)
        {
            double yi = counts[lev[j]];
            mat x1 = sigmagw[j].row(i);
            sigmai = sigmai + yi * x1;
            yisum += yi;
        }
        sigma1gw.row(i) = sigmai / yisum;
    }

    mat logPf = mat(nPr, m, fill::zeros);
    for (uword i = 0; i < m; i++)
    {
        for (uword j = 0; j < nPr; j++)
        {
            vec xprj = xpr.row(j).t();
            vec meani = localMean[i].row(j).t();
            mat covmatj = sigma1gw.row(j);
            vec x1 = 0.5 * (xprj - meani).t() * solve(covmatj, (xprj - meani));
            logPf(j, i) = (m / 2) * log(norm(covmatj)) + x1(0) - log(prior[i].at(j));
        }
    }
    vector<string> groupPr;
    for (uword i = 0; i < nPr; i++)
    {
        uword index = index_min(logPf.row(i));
        groupPr.push_back((lev[index]));
    }
    mGroup = groupPr;
    return logPf;
}

// template<class T>
mat GWDA::wMean(arma::mat &x, arma::mat &wt)
{
    uword nVar = x.n_cols;
    uword nPr = wt.n_cols;
    mat localMean = mat(nPr, nVar, fill::zeros);
    for (uword i = 0; i < nPr; i++)
    {
        rowvec w = wt.col(i).t();
        double sumw = sum(w);
        rowvec wi = w / sumw;
        localMean.row(i) = wi * x;
    }
    return localMean;
}

// template<class T>
cube GWDA::wVarCov(arma::mat &x, arma::mat &wt)
{
    uword nVar = x.n_cols;
    uword nPr = wt.n_cols;
    cube Covmat(nPr, nVar, nVar, fill::zeros);
    for (uword i = 0; i < nPr; i++)
    {
        vec w = wt.col(i);
        double sumw = sum(w);
        vec wi = w / sumw;
        if (nVar >= 2)
        {
            Covmat.row(i) = covwtmat(x, wi);
        }
    }
    return Covmat;
}

// template<class T>
vec GWDA::wPrior(arma::mat &wt, double sumW)
{
    uword nPr = wt.n_cols;
    vec localPrior = vec(nPr, fill::zeros);
    for (uword i = 0; i < nPr; i++)
    {
        localPrior(i) = sum(wt.col(i)) / sumW;
    }
    return localPrior;
}

// template<class T>
void GWDA::setParallelType(const ParallelType &type)
{
    if (type & parallelAbility())
    {
        mParallelType = type;
        switch (type)
        {
        case ParallelType::SerialOnly:
            mDiscriminantAnalysisFunction = &GWDA::discriminantAnalysisSerial;
            break;
#ifdef ENABLE_OPENMP
        case ParallelType::OpenMP:
            mDiscriminantAnalysisFunction = &GWDA::discriminantAnalysisOmp;
            break;
#endif
        default:
            mDiscriminantAnalysisFunction = &GWDA::discriminantAnalysisSerial;
            break;
        }
    }
}