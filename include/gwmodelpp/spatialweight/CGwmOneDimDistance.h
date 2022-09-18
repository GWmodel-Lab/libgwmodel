#ifndef CGWMOneDimDISTANCE_H
#define CGWMOneDimDISTANCE_H

#include "CGwmDistance.h"

/**
 * @brief Struct of parameters used in spatial distance calculating according to coordinate reference system. 
 * Usually a pointer to object of this class is passed to CGwmOneDimDistance::distance().
 */
struct OneDimDistanceParameter : public DistanceParameter
{
    vec focusPoints;    //!< Matrix of focus points' coordinates. The shape of it must be nx2 and the first column is longitudes or x-coordinate, the second column is latitudes or y-coordinate.
    vec dataPoints;     //!< Matrix of data points' coordinates. The shape of it must be nx2 and the first column is longitudes or x-coordinate, the second column is latitudes or y-coordinate.

    /**
     * @brief Construct a new OneDimDistanceParameter object.
     * 
     * @param fp Reference to focus points.
     * @param dp Reference to data points.
     */
    OneDimDistanceParameter(const vec& fp, const vec& dp) : DistanceParameter()
        , focusPoints(fp)
        , dataPoints(dp)
    {
        total = fp.n_rows;
    }
};

/**
 * @brief Class for calculating spatial distance according to coordinate reference system.
 */
class CGwmOneDimDistance : public CGwmDistance
{
public:

    /**
     * @brief Calculate spatial distance for points with geographical coordinate reference system.
     * 
     * @param out_loc Matrix of focus point' coordinate. The shape of it must be 1x2 and the first column is longitudes, the second column is latitudes.
     * @param in_locs Matrix of data points' coordinates. The shape of it must be nx2 and the first column is longitudes, the second column is latitudes.
     * @return vec Distance vector for out_loc.
     */
    static vec AbstractDistance(const double& out_loc, const vec& in_locs)
    {
        // vec d = abs(in_locs - out_loc);
        // d.print("d");
        return abs(in_locs - out_loc);
    }

public:

    CGwmOneDimDistance();

    /**
     * @brief Construct a new CGwmOneDimDistance object
     * 
     * @param isGeographic Whether the coordinate reference system is geographical.
     */
    explicit CGwmOneDimDistance(bool isGeographic);

    /**
     * @brief Construct a new CGwmOneDimDistance object.
     * 
     * @param distance Refernce to object for copying.
     */
    CGwmOneDimDistance(const CGwmOneDimDistance& distance);

    virtual CGwmDistance * clone() override
    {
        return new CGwmOneDimDistance(*this);
    }

    DistanceType type() override { return DistanceType::OneDimDistance; }

public:
    virtual vec distance(DistanceParameter* parameter, uword focus) override;
};

#endif // CGWMOneDimDISTANCE_H
