#ifndef CGWMDMATDISTANCE_H
#define CGWMDMATDISTANCE_H

#include <string>
#include "CGwmDistance.h"


/**
 * [NOT AVALIABLE]
 */
class CGwmDMatDistance : public CGwmDistance
{
public:

    struct Parameter : public CGwmDistance::Parameter
    {
        arma::uword rowSize;

        Parameter(arma::uword size, arma::uword rows) : rowSize(size) 
        {
            total = rows;
        }
    };

public:
    explicit CGwmDMatDistance(std::string dmatFile);
    CGwmDMatDistance(const CGwmDMatDistance& distance);

    virtual CGwmDistance * clone() const override
    {
        return new CGwmDMatDistance(*this);
    }

    DistanceType type() const override { return DistanceType::DMatDistance; }

    std::string dMatFile() const;
    void setDMatFile(const std::string &dMatFile);

public:

    /**
     * @brief Create Parameter for Caclulating CRS Distance.
     * 
     * @param plist A list of parameters containing 2 items:
     *  - `arma::uword` size
     *  - `arma::uword` rows
     *  . 
     * 
     * @return DistanceParameter* The pointer to parameters.
     */
    virtual void makeParameter(std::initializer_list<DistParamVariant> plist) override;
    
    virtual arma::vec distance(arma::uword focus) const override;
    virtual double maxDistance() const override;
    virtual double minDistance() const override;

private:
    std::string mDMatFile;
    std::unique_ptr<Parameter> mParameter = nullptr;
};

inline std::string CGwmDMatDistance::dMatFile() const
{
    return mDMatFile;
}

inline void CGwmDMatDistance::setDMatFile(const std::string &dMatFile)
{
    mDMatFile = dMatFile;
}

#endif // CGWMDMATDISTANCE_H
