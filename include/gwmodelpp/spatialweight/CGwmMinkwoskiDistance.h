#ifndef CGWMMINKWOSKIDISTANCE_H
#define CGWMMINKWOSKIDISTANCE_H

#include "CGwmCRSDistance.h"

class CGwmMinkwoskiDistance : public CGwmCRSDistance
{
public:
    static arma::mat CoordinateRotate(const arma::mat& coords, double theta);
    static arma::vec ChessDistance(const arma::rowvec& out_loc, const arma::mat& in_locs);
    static arma::vec ManhattonDistance(const arma::rowvec& out_loc, const arma::mat& in_locs);
    static arma::vec MinkwoskiDistance(const arma::rowvec& out_loc, const arma::mat& in_locs, double p);

public:
    explicit CGwmMinkwoskiDistance(double p, double theta);
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
    virtual arma::vec distance(arma::uword focus) override;

private:
    double mPoly;
    double mTheta;
};

inline arma::vec CGwmMinkwoskiDistance::ChessDistance(const arma::rowvec& out_loc, const arma::mat& in_locs)
{
    return max(abs(in_locs.each_row() - out_loc), 1);
}

inline arma::vec CGwmMinkwoskiDistance::ManhattonDistance(const arma::rowvec& out_loc, const arma::mat& in_locs)
{
    return sum(abs(in_locs.each_row() - out_loc), 1);
}

inline arma::vec CGwmMinkwoskiDistance::MinkwoskiDistance(const arma::rowvec& out_loc, const arma::mat& in_locs, double p)
{
    arma::vec temp = abs(in_locs.each_row() - out_loc);
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
