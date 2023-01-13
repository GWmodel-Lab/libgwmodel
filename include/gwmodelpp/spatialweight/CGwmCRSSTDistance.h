#ifndef CGWMCRSSTDISTANCE_H
#define CGWMCRSSTDISTANCE_H

#include "CGwmCRSDistance.h"

class CGwmCRSSTDistance : public CGwmCRSDistance
{

    /**
     * @brief Struct of parameters used in spatial distance calculating according to coordinate reference system. 
     * Usually a pointer to object of this class is passed to CGwmCRSSTDistance::distance().
     */
        struct Parameter : public CGwmDistance::Parameter
    {
        mat focusPoints;    //!< Matrix of focus points' coordinates. The shape of it must be nx3 and the first column is longitudes or x-coordinate, 
                            //the second column is latitudes or y-coordinate, the third column is time index;
        mat dataPoints;     //!< Matrix of data points' coordinates. The shape of it must be nx3 and the first column is longitudes or x-coordinate,
                            //the second column is latitudes or y-coordinate, the third column is time index;
        /**
         * @brief Construct a new CRSDistanceParameter object.
         * 
         * @param fp Reference to focus points.
         * @param dp Reference to data points.
         */
        Parameter(const mat& fp, const mat& dp) : CGwmDistance::Parameter()
            , focusPoints(fp)
            , dataPoints(dp)
        {
            total = fp.n_rows;
        }
    };


public:
    static vec GcrsSTDistance(const rowvec& out_loc, const mat& in_locs ,double mLambda);

    static vec EuclideanDistance(const rowvec& out_loc, const mat& in_locs ,double mLambda);


public:
    CGwmCRSSTDistance();

    explicit CGwmCRSSTDistance(bool isGeographic, double lambda);    

    /**
     * @brief Copy construct a new CGwmCRSDistance object.
     * 
     * @param distance Refernce to object for copying.
     */
    CGwmCRSSTDistance(const CGwmCRSSTDistance& distance);

    virtual CGwmDistance * clone() override
    {
        return new CGwmCRSSTDistance(*this);
    }


    DistanceType type() override { return DistanceType::CRSSTDistance; }

public:

    /**
     * @brief Create Parameter for Caclulating CRS Distance.
     * 
     * @param plist A list of parameters containing 2 items:
     *  - `mat` focus points
     *  - `mat` data points
     *  .
     * 
     * @return DistanceParameter* The pointer to parameters.
     */
    virtual void makeParameter(initializer_list<DistParamVariant> plist) override;

    virtual vec distance(uword focus) override;

    double maxDistance();

protected:

    unique_ptr<Parameter> mParameter;
    bool mGeographic;
    double mLambda=0.0;

private:
    typedef vec (*CalculatorType)(const rowvec&, const mat&, const double);
private:
    CalculatorType mCalculator = &EuclideanDistance;
    //CalculatorType mCalculator;
};


#endif // CGWMCRSSTDISTANCE_H