#ifndef CGWMOneDimDISTANCE_H
#define CGWMOneDimDISTANCE_H

#include "Distance.h"

namespace gwm
{

/**
 * @brief Class for calculating spatial distance according to coordinate reference system.
 */
class OneDimDistance : public Distance
{
public:

    /**
     * @brief Struct of parameters used in spatial distance calculating according to coordinate reference system. 
     * Usually a pointer to object of this class is passed to OneDimDistance::distance().
     */
    struct Parameter : public Distance::Parameter
    {
        arma::vec focusPoints;    //!< Matrix of focus points' coordinates. The shape of it must be nx2 and the first column is longitudes or x-coordinate, the second column is latitudes or y-coordinate.
        arma::vec dataPoints;     //!< Matrix of data points' coordinates. The shape of it must be nx2 and the first column is longitudes or x-coordinate, the second column is latitudes or y-coordinate.

        /**
         * @brief Construct a new OneDimDistanceParameter object.
         * 
         * @param fp Reference to focus points.
         * @param dp Reference to data points.
         */
        Parameter(const arma::vec& fp, const arma::vec& dp) : Distance::Parameter()
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
    static arma::vec AbstractDistance(const double& out_loc, const arma::vec& in_locs)
    {
        // arma::vec d = abs(in_locs - out_loc);
        // d.print("d");
        return abs(in_locs - out_loc);
    }

public:

    OneDimDistance();

    /**
     * @brief Construct a new OneDimDistance object
     * 
     * @param isGeographic Whether the coordinate reference system is geographical.
     */
    explicit OneDimDistance(bool isGeographic);

    /**
     * @brief Construct a new OneDimDistance object.
     * 
     * @param distance Refernce to object for copying.
     */
    OneDimDistance(const OneDimDistance& distance);

    virtual Distance * clone() override
    {
        return new OneDimDistance(*this);
    }

    DistanceType type() override { return DistanceType::OneDimDistance; }

public:

    /**
     * @brief Create Parameter for Caclulating CRS Distance.
     * 
     * @param plist A list of parameters containing 2 items:
     *  - `arma::vec` focus coordinates (one column)
     *  - `arma::vec` data coordinates (one column)
     *  .
     * 
     * @return DistanceParameter* The pointer to parameters.
     */
    virtual void makeParameter(std::initializer_list<DistParamVariant> plist) override;

    virtual arma::vec distance(arma::uword focus) override;
    virtual double maxDistance() override;
    virtual double minDistance() override;

protected:
    std::unique_ptr<Parameter> mParameter = nullptr;
};

}

#endif // CGWMOneDimDISTANCE_H
