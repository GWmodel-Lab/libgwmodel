#ifndef CGWMDISTANCE_H
#define CGWMDISTANCE_H

#include <unordered_map>
#include <string>
#include <armadillo>
#include <variant>

using namespace std;
using namespace arma;

/**
 * @brief Struct of parameters used in spatial distance calculating. 
 * Usually a pointer to object of its derived classes is passed to CGwmDistance::distance().
 */
struct DistanceParameter
{
    uword total;    //!< Total focus points.

    /**
     * @brief Construct a new DistanceParameter object.
     */
    DistanceParameter(): total(0) {}
};

typedef variant<mat, vec, uword> DistParamVariant;

/**
 * @brief Abstract base class for calculating spatial distance.
 * It defines some interface commonly used in spatial algorithms:
 * 
 * - Clone this object.
 * - Calculate distance vector for a focus point.
 * - Get maximum distance among all points.
 * - Get minimum distance among all points.
 * 
 * Pointer of this type can be put to these classes:
 * 
 * - CGwmCRSDistance
 * - CGwmMinkwoskiDistance
 * 
 */
class CGwmDistance
{
public:

    /**
     * @brief Enum for types of distance.
     */
    enum DistanceType
    {
        CRSDistance,        //!< Distance according to coordinate reference system.
        MinkwoskiDistance,  //!< Minkwoski distance
        DMatDistance        //!< Distance according to a .dmat file
    };
    
    /**
     * @brief A mapper between types of distance and its names.
     * 
     */
    static unordered_map<DistanceType, string> TypeNameMapper;

public:

    /**
     * @brief Construct a new CGwmDistance object.
     */
    explicit CGwmDistance() {};

    /**
     * @brief Construct a new CGwmDistance object.
     * 
     * @param d Reference to object for copying.
     */
    CGwmDistance(const CGwmDistance& d) {};

    /**
     * @brief Destroy the CGwmDistance object.
     */
    virtual ~CGwmDistance() {};

    /**
     * @brief Clone this CGwmDistance object.
     * 
     * @return Newly created pointer.
     */
    virtual CGwmDistance* clone() = 0;

    /**
     * @brief Return the type of this object.
     * 
     * @return Type of distance. 
     */
    virtual DistanceType type() = 0;


public:

    virtual DistanceParameter* makeParameter(initializer_list<DistParamVariant> plist) = 0;

    /**
     * @brief Calculate distance vector for a focus point. 
     * 
     * @param parameter Pointer to parameter object used for calculating distance. 
     * @param focus Focused point's index. Require focus < total.
     * @return Distance vector for the focused point.
     */
    virtual vec distance(DistanceParameter* parameter, uword focus) = 0;

    /**
     * @brief Get maximum distance among all points。
     * 
     * @param total Total number of points.
     * @param parameter Pointer to parameter object used for calculating distance. 
     * @return Maximum distance. 
     */
    double maxDistance(uword total, DistanceParameter* parameter);
    
    /**
     * @brief Get minimum distance among all points
     * 
     * @param total Total number of points.
     * @param parameter Pointer to parameter object used for calculating distance. 
     * @return Maximum distance.  
     */
    double minDistance(uword total, DistanceParameter* parameter);
};


#endif // CGWMDISTANCE_H
