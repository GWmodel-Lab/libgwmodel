#ifndef CGWMMINKWOSKIDISTANCE_H
#define CGWMMINKWOSKIDISTANCE_H

#include "spatialweight/CGwmCRSDistance.h"

class CGwmMinkwoskiDistance : public CGwmCRSDistance
{
public:
    static mat CoordinateRotate(const mat& coords, double theta);
    static vec ChessDistance(const rowvec& out_loc, const mat& in_locs);
    static vec ManhattonDistance(const rowvec& out_loc, const mat& in_locs);
    static vec MinkwoskiDistance(const rowvec& out_loc, const mat& in_locs, double p);

public:
    explicit CGwmMinkwoskiDistance(int total, double p, double theta);
    CGwmMinkwoskiDistance(const CGwmMinkwoskiDistance& distance);

    virtual CGwmDistance * clone() override
    {
        return new CGwmMinkwoskiDistance(*this);
    }

    DistanceType type() override { return DistanceType::MinkwoskiDistance; }

    double poly() const;
    void setPoly(double poly);

    double theta() const;
    void setTheta(double theta);

public:
    virtual vec distance(int focus) override;

private:
    double mPoly;
    double mTheta;
};

inline vec CGwmMinkwoskiDistance::ChessDistance(const rowvec& out_loc, const mat& in_locs)
{
    return max(abs(in_locs.each_row() - out_loc), 1);
}

inline vec CGwmMinkwoskiDistance::ManhattonDistance(const rowvec& out_loc, const mat& in_locs)
{
    return sum(abs(in_locs.each_row() - out_loc), 1);
}

inline vec CGwmMinkwoskiDistance::MinkwoskiDistance(const rowvec& out_loc, const mat& in_locs, double p)
{
    vec temp = abs(in_locs.each_row() - out_loc);
    return pow(sum(pow(temp, p), 1), 1.0 / p);
}

inline double CGwmMinkwoskiDistance::poly() const
{
    return mPoly;
}

inline void CGwmMinkwoskiDistance::setPoly(double poly)
{
    mPoly = poly;
}

inline double CGwmMinkwoskiDistance::theta() const
{
    return mTheta;
}

inline void CGwmMinkwoskiDistance::setTheta(double theta)
{
    mTheta = theta;
}

#endif // CGWMMINKWOSKIDISTANCE_H
