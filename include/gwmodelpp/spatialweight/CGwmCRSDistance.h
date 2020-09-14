#ifndef CGWMCRSDISTANCE_H
#define CGWMCRSDISTANCE_H

#include "spatialweight/CGwmDistance.h"

class CGwmCRSDistance : public CGwmDistance
{
public:
    static vec SpatialDistance(const rowvec& out_loc, const mat& in_locs);
    static vec EuclideanDistance(const rowvec& out_loc, const mat& in_locs);
    static double SpGcdist(double lon1, double lon2, double lat1, double lat2);

public:
    explicit CGwmCRSDistance(int total, bool isGeographic);
    CGwmCRSDistance(const CGwmCRSDistance& distance);

    virtual CGwmDistance * clone() override
    {
        return new CGwmCRSDistance(*this);
    }

    DistanceType type() override { return DistanceType::CRSDistance; }

    bool geographic() const;
    void setGeographic(bool geographic);

    mat *focusPoints() const;
    void setFocusPoints(mat *focusPoints);

    mat *dataPoints() const;
    void setDataPoints(mat *dataPoints);

public:
    virtual vec distance(int focus) override;
    uword length() const override;

protected:
    bool mGeographic = false;
    mat* mFocusPoints = nullptr;
    mat* mDataPoints = nullptr;
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
}

inline mat *CGwmCRSDistance::focusPoints() const
{
    return mFocusPoints;
}

inline void CGwmCRSDistance::setFocusPoints(mat *focusPoints)
{
    mFocusPoints = focusPoints;
    mTotal = focusPoints->n_rows;
}

inline mat *CGwmCRSDistance::dataPoints() const
{
    return mDataPoints;
}

inline void CGwmCRSDistance::setDataPoints(mat *dataPoints)
{
    mDataPoints = dataPoints;
}

inline uword CGwmCRSDistance::length() const
{
    return mDataPoints->n_rows;
}

#endif // CGWMCRSDISTANCE_H
