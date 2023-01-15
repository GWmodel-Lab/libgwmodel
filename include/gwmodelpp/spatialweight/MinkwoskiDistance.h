#ifndef MINKWOSKIDISTANCE_H
#define MINKWOSKIDISTANCE_H

#include "CRSDistance.h"

namespace gwm
{

class MinkwoskiDistance : public CRSDistance
{
public:
    static arma::mat CoordinateRotate(const arma::mat& coords, double theta);
    static arma::vec ChessDistance(const arma::rowvec& out_loc, const arma::mat& in_locs);
    static arma::vec ManhattonDist(const arma::rowvec& out_loc, const arma::mat& in_locs);
    static arma::vec MinkwoskiDist(const arma::rowvec& out_loc, const arma::mat& in_locs, double p);

public:
    explicit MinkwoskiDistance(double p, double theta);
    MinkwoskiDistance(const MinkwoskiDistance& distance);

    virtual Distance * clone() override
    {
        return new MinkwoskiDistance(*this);
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

inline arma::vec MinkwoskiDistance::ChessDistance(const arma::rowvec& out_loc, const arma::mat& in_locs)
{
    return max(abs(in_locs.each_row() - out_loc), 1);
}

inline arma::vec MinkwoskiDistance::ManhattonDist(const arma::rowvec& out_loc, const arma::mat& in_locs)
{
    return sum(abs(in_locs.each_row() - out_loc), 1);
}

inline arma::vec MinkwoskiDistance::MinkwoskiDist(const arma::rowvec& out_loc, const arma::mat& in_locs, double p)
{
    arma::vec temp = abs(in_locs.each_row() - out_loc);
    return pow(sum(pow(temp, p), 1), 1.0 / p);
}

inline double MinkwoskiDistance::poly() const
{
    return mPoly;
}

inline void MinkwoskiDistance::setPoly(double poly)
{
    mPoly = poly;
}

inline double MinkwoskiDistance::theta() const
{
    return mTheta;
}

inline void MinkwoskiDistance::setTheta(double theta)
{
    mTheta = theta;
}

}

#endif // MINKWOSKIDISTANCE_H
