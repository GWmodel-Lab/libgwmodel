#ifndef CGWMCRSDISTANCE_H
#define CGWMCRSDISTANCE_H

#include "spatialweight/CGwmDistance.h"

struct CRSDistanceParameter : public DistanceParameter
{
    const mat& focusPoints;
    const mat& dataPoints;

    CRSDistanceParameter(const mat& fp, const mat& dp) : DistanceParameter()
        , focusPoints(fp)
        , dataPoints(dp)
    {
    }
};

class CGwmCRSDistance : public CGwmDistance
{
public:
    static vec SpatialDistance(const rowvec& out_loc, const mat& in_locs);
    static vec EuclideanDistance(const rowvec& out_loc, const mat& in_locs);
    static double SpGcdist(double lon1, double lon2, double lat1, double lat2);

private:
    typedef vec (*CalculatorType)(const rowvec&, const mat&);

public:
    explicit CGwmCRSDistance(bool isGeographic);
    CGwmCRSDistance(const CGwmCRSDistance& distance);

    virtual CGwmDistance * clone() override
    {
        return new CGwmCRSDistance(*this);
    }

    DistanceType type() override { return DistanceType::CRSDistance; }

    bool geographic() const;
    void setGeographic(bool geographic);

public:
    virtual vec distance(DistanceParameter* parameter) override;

protected:
    bool mGeographic = false;

private:
    CalculatorType mCalculator = &EuclideanDistance;
};

inline vec CGwmCRSDistance::EuclideanDistance(const rowvec& out_loc, const mat& in_locs)
{
    mat diff = (in_locs.each_row() - out_loc);
    return sqrt(sum(diff % diff, 1));
}

inline bool CGwmCRSDistance::geographic() const
{
    return mGeographic;
}

inline void CGwmCRSDistance::setGeographic(bool geographic)
{
    mGeographic = geographic;
    mCalculator = mGeographic ? &SpatialDistance : &EuclideanDistance;
}

#endif // CGWMCRSDISTANCE_H
