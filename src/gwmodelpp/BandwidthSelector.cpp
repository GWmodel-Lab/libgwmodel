#include "BandwidthSelector.h"
#include <iostream>
#include <iomanip>

using namespace std;
using namespace gwm;

Status BandwidthSelector::optimize(IBandwidthSelectable* instance)
{
    cerr << "[BandwidthSelector] optimize() starting: lower=" << mLower << ", upper=" << mUpper
         << ", adaptive=" << std::boolalpha << mBandwidth->adaptive() << ", kernel=" << mBandwidth->kernel() << "\n";
    BandwidthWeight* w1 = static_cast<BandwidthWeight*>(mBandwidth->clone());
    BandwidthWeight* w2 = static_cast<BandwidthWeight*>(mBandwidth->clone());
    double xU = mUpper, xL = mLower;
    bool adaptBw = mBandwidth.adaptive();
    const double eps = 1e-4;
    const double R = (sqrt(5)-1)/2;
    int iter = 0;
    double d = R * (xU - xL);
    double x1 = adaptBw ? floor(xL + d) : (xL + d);
    double x2 = adaptBw ? round(xU - d) : (xU - d);
    w1->setBandwidth(x1);
    w2->setBandwidth(x2);
    double f1 = DBL_MAX, f2 = DBL_MAX;
    Status s1 = instance->getCriterion(w1, f1);
    Status s2 = instance->getCriterion(w2, f2);
    if (s1 == Status::Terminated || s2 == Status::Terminated)
    {
        return Status::Terminated;
    }
    if (f1 == DBL_MAX && f2 == DBL_MAX)
    {
        throw std::runtime_error("Invalid initial values.");
    }
    if (f1 < DBL_MAX)
        mBandwidthCriterion[x1] = f1;
    if (f2 < DBL_MAX)
        mBandwidthCriterion[x2] = f2;
    cerr << fixed << setprecision(6);
    cerr << "[BandwidthSelector] initial x1=" << x1 << ", f1=" << f1 << "; x2=" << x2 << ", f2=" << f2 << "\n";
    double d1 = f2 - f1;
    double xopt = f1 < f2 ? x1 : x2;
    double ea = 100;
    while ((s1 == Status::Success) && (s2 == Status::Success) && (fabs(d) > eps) && (fabs(d1) > eps) && iter < ea)
    {
        d = R * d;
        if (f1 < f2)
        {
            xL = x2;
            x2 = x1;
            x1 = adaptBw ? round(xL + d) : (xL + d);
            f2 = f1;
            w1->setBandwidth(x1);
            s1 = instance->getCriterion(w1, f1);
            if (f1 < DBL_MAX)
            mBandwidthCriterion[x1] = f1;
            cerr << "[BandwidthSelector] iter=" << iter << " x1=" << x1 << " f1=" << f1 << " x2=" << x2 << " f2=" << f2 << " xopt=" << xopt << " d=" << d << "\n";
        }
        else
        {
            xU = x1;
            x1 = x2;
            x2 = adaptBw ? floor(xU - d) : (xU - d);
            f1 = f2;
            w2->setBandwidth(x2);
            s2 = instance->getCriterion(w2, f2);
            if (f2 < DBL_MAX)
                mBandwidthCriterion[x2] = f2;
        }
        iter = iter + 1;
        xopt = (f1 < f2) ? x1 : x2;
        d1 = f2 - f1;
        cerr << "[BandwidthSelector] iter=" << iter << " x1=" << x1 << " f1=" << f1 << " x2=" << x2 << " f2=" << f2 << " xopt=" << xopt << " d=" << d << "\n";
    }
    delete w1;
    delete w2;
    cerr << "[BandwidthSelector] optimize completed: xopt=" << xopt << ", s1=" << static_cast<int>(s1) << ", s2=" << static_cast<int>(s2) << "\n";
    if (s1 == Status::Success && s2 == Status::Success)
    {
        mOptimisedBandwidth.setBandwidth(xopt);
        return Status::Success;
    }
    else
    {
        return Status::Success;
    }
}

BandwidthCriterionList BandwidthSelector::bandwidthCriterion() const
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