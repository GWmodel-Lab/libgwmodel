#include "CGwmBandwidthSelector.h"

CGwmBandwidthSelector::CGwmBandwidthSelector()
{
}

CGwmBandwidthSelector::CGwmBandwidthSelector(CGwmBandwidthWeight* bandwidth, double lower, double upper)
    : mBandwidth(bandwidth)
    , mLower(lower)
    , mUpper(upper)
{
}

CGwmBandwidthSelector::~CGwmBandwidthSelector()
{
}

CGwmBandwidthWeight* CGwmBandwidthSelector::optimize(IGwmBandwidthSelectable* instance)
{
    CGwmBandwidthWeight* w1 = static_cast<CGwmBandwidthWeight*>(mBandwidth->clone());
    CGwmBandwidthWeight* w2 = static_cast<CGwmBandwidthWeight*>(mBandwidth->clone());
    double xU = mUpper, xL = mLower;
    bool adaptBw = mBandwidth->adaptive();
    const double eps = 1e-4;
    const double R = (sqrt(5)-1)/2;
    int iter = 0;
    double d = R * (xU - xL);
    double x1 = adaptBw ? floor(xL + d) : (xL + d);
    double x2 = adaptBw ? round(xU - d) : (xU - d);
    w1->setBandwidth(x1);
    w2->setBandwidth(x2);
    double f1 = instance->getCriterion(w1);
    double f2 = instance->getCriterion(w2);
    if (f1 < DBL_MAX)
        mBandwidthCriterion[x1] = f1;
    if (f2 < DBL_MAX)
        mBandwidthCriterion[x2] = f2;
    double d1 = f2 - f1;
    double xopt = f1 < f2 ? x1 : x2;
    double ea = 100;
    while ((fabs(d) > eps) && (fabs(d1) > eps) && iter < ea)
    {
        d = R * d;
        if (f1 < f2)
        {
            xL = x2;
            x2 = x1;
            x1 = adaptBw ? round(xL + d) : (xL + d);
            f2 = f1;
            w1->setBandwidth(x1);
            f1 = instance->getCriterion(w1);
            if (f1 < DBL_MAX)
                mBandwidthCriterion[x1] = f1;
        }
        else
        {
            xU = x1;
            x1 = x2;
            x2 = adaptBw ? floor(xU - d) : (xU - d);
            f1 = f2;
            w2->setBandwidth(x2);
            f2 = instance->getCriterion(w2);
            if (f2 < DBL_MAX)
                mBandwidthCriterion[x2] = f2;
        }
        iter = iter + 1;
        xopt = (f1 < f2) ? x1 : x2;
        d1 = f2 - f1;
    }
    delete w1;
    delete w2;
    CGwmBandwidthWeight* wopt = new CGwmBandwidthWeight();
    wopt->setKernel(mBandwidth->kernel());
    wopt->setAdaptive(mBandwidth->adaptive());
    wopt->setBandwidth(xopt);
    return wopt;
}

BandwidthCriterionList CGwmBandwidthSelector::bandwidthCriterion() const
{
    BandwidthCriterionList criterions;
    for (auto item : mBandwidthCriterion)
    {
        criterions.push_back(make_pair(item.first, item.second));
    }
    std::sort(criterions.begin(), criterions.end(), [](const pair<double, double>& a, const pair<double, double>& b)
    {
        return a.first < b.first;
    });
    return criterions;
}