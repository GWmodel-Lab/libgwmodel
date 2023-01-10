#ifndef CGWMDISTANCE_H
#define CGWMDISTANCE_H

#include <memory>
#include <string>
#include <unordered_map>
#include <armadillo>
#include <variant>


namespace gwm
{

typedef std::variant<arma::mat, arma::vec, arma::uword> DistParamVariant;

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
     * @brief Struct of parameters used in spatial distance calculating. 
     * Usually a pointer to object of its derived classes is passed to CGwmDistance::distance().
     */
    struct Parameter
    {
        arma::uword total;    //!< Total focus points.

        /**
         * @brief Construct a new DistanceParameter object.
         */
        Parameter(): total(0) {}
    };

    /**
     * @brief Enum for types of distance.
     */
    enum DistanceType
    {
        CRSDistance,        //!< Distance according to coordinate reference system.
        MinkwoskiDistance,  //!< Minkwoski distance
        DMatDistance,       //!< Distance according to a .dmat file
        OneDimDistance,     //!< Distance for just one dimension
    };
    
    /**
     * @brief A mapper between types of distance and its names.
     * 
     */
    static std::unordered_map<DistanceType, std::string> TypeNameMapper;

public:

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

    virtual Parameter* parameter() const = delete;


public:

    /**
     * @brief Create Parameter for Caclulating Distance.
     * This function is pure virtual. It would never be called directly.
     * 
     * @param plist A list of parameters. 
     */
    virtual void makeParameter(std::initializer_list<DistParamVariant> plist) = 0;

    /**
     * @brief Calculate distance vector for a focus point. 
     * 
     * @param parameter Pointer to parameter object used for calculating distance. 
     * @param focus Focused point's index. Require focus < total.
     * @return Distance vector for the focused point.
     */
    virtual arma::vec distance(arma::uword focus) = 0;

    /**
     * @brief Get maximum distance among all points。
     * 
     * @param total Total number of points.
     * @param parameter Pointer to parameter object used for calculating distance. 
     * @return Maximum distance. 
     */
    virtual double maxDistance() = 0;
    
    /**
     * @brief Get minimum distance among all points
     * 
     * @param total Total number of points.
     * @param parameter Pointer to parameter object used for calculating distance. 
     * @return Maximum distance.  
     */
    virtual double minDistance() = 0;

};

}


#endif // CGWMDISTANCE_H
