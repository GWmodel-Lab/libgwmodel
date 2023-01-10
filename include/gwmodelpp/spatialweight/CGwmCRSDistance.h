#ifndef CGWMCRSDISTANCE_H
#define CGWMCRSDISTANCE_H

#include "CGwmDistance.h"

/**
 * @brief Class for calculating spatial distance according to coordinate reference system.
 */
class CGwmCRSDistance : public CGwmDistance
{
public:

    /**
     * @brief Struct of parameters used in spatial distance calculating according to coordinate reference system. 
     * Usually a pointer to object of this class is passed to CGwmCRSDistance::distance().
     */
    struct Parameter : public CGwmDistance::Parameter
    {
        arma::mat focusPoints;    //!< Matrix of focus points' coordinates. The shape of it must be nx2 and the first column is longitudes or x-coordinate, the second column is latitudes or y-coordinate.
        arma::mat dataPoints;     //!< Matrix of data points' coordinates. The shape of it must be nx2 and the first column is longitudes or x-coordinate, the second column is latitudes or y-coordinate.

        /**
         * @brief Construct a new CRSDistanceParameter object.
         * 
         * @param fp Reference to focus points.
         * @param dp Reference to data points.
         */
        Parameter(const arma::mat& fp, const arma::mat& dp) : CGwmDistance::Parameter()
            , focusPoints(fp)
            , dataPoints(dp)
        {
            total = fp.n_rows;
        }
    };

public:

    /**
     * @brief Calculate spatial distance for points with geographical coordinate reference system.
     * 
     * @param out_loc Matrix of focus point' coordinate. The shape of it must be 1x2 and the first column is longitudes, the second column is latitudes.
     * @param in_locs Matrix of data points' coordinates. The shape of it must be nx2 and the first column is longitudes, the second column is latitudes.
     * @return arma::vec Distance vector for out_loc.
     */
    static arma::vec SpatialDistance(const arma::rowvec& out_loc, const arma::mat& in_locs);

    /**
     * @brief Calculate euclidean distance for points with geographical coordinate reference system.
     * 
     * @param out_loc Matrix of focus point' coordinate. The shape of it must be 1x2 and the first column is x-coordinate, the second column is y-coordinate.
     * @param in_locs Matrix of data points' coordinates. The shape of it must be nx2 and the first column is x-coordinate, the second column is y-coordinate.
     * @return arma::vec Distance vector for out_loc.
     */
    static arma::vec EuclideanDistance(const arma::rowvec& out_loc, const arma::mat& in_locs)
    {
        arma::mat diff = (in_locs.each_row() - out_loc);
        return sqrt(sum(diff % diff, 1));
    }

    /**
     * @brief Calculate spatial distance for two points with geographical coordinate reference system.
     * 
     * @param lon1 Longitude of point 1.
     * @param lon2 Longitude of point 2.
     * @param lat1 Latitude of point 1.
     * @param lat2 Latitude of point 2.
     * @return double Spatial distance for point 1 and point 2.
     */
    static double SpGcdist(double lon1, double lon2, double lat1, double lat2);

private:
    typedef arma::vec (*CalculatorType)(const arma::rowvec&, const arma::mat&);

public:

    CGwmCRSDistance() : mGeographic(false), mParameter(nullptr) {}

    /**
     * @brief Construct a new CGwmCRSDistance object
     * 
     * @param isGeographic Whether the coordinate reference system is geographical.
     */
    explicit CGwmCRSDistance(bool isGeographic): mGeographic(isGeographic), mParameter(nullptr)
    {
        mCalculator = mGeographic ? &SpatialDistance : &EuclideanDistance;
    }

    /**
     * @brief Copy construct a new CGwmCRSDistance object.
     * 
     * @param distance Refernce to object for copying.
     */
    CGwmCRSDistance(const CGwmCRSDistance& distance);

    virtual CGwmDistance * clone() override
    {
        return new CGwmCRSDistance(*this);
    }

    DistanceType type() override { return DistanceType::CRSDistance; }

    /**
     * @brief Get the CGwmCRSDistance::mGeographic object.
     * 
     * @return true if the coordinate reference system is geographical.
     * @return false if the coordinate reference system is not geographical.
     */
    bool geographic() const
    {
        return mGeographic;
    }

    /**
     * @brief Set the CGwmCRSDistance::mGeographic object.
     * 
     * @param geographic Whether the coordinate reference system is geographical.
     */
    void setGeographic(bool geographic)
    {
        mGeographic = geographic;
        mCalculator = mGeographic ? &SpatialDistance : &EuclideanDistance;
    }

public:

    /**
     * @brief Create Parameter for Caclulating CRS Distance.
     * 
     * @param plist A list of parameters containing 2 items:
     *  - `arma::mat` focus points
     *  - `arma::mat` data points
     *  .
     * 
     * @return DistanceParameter* The pointer to parameters.
     */
    virtual void makeParameter(std::initializer_list<DistParamVariant> plist) override;

    virtual arma::vec distance(arma::uword focus) override;
    virtual double maxDistance() override;
    virtual double minDistance() override;

protected:
    bool mGeographic;
    std::unique_ptr<Parameter> mParameter;

private:
    CalculatorType mCalculator = &EuclideanDistance;
};

#endif // CGWMCRSDISTANCE_H
