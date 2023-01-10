#ifndef DMATDISTANCE_H
#define DMATDISTANCE_H

#include <string>
#include "Distance.h"


namespace gwm
{

/**
 * [NOT AVALIABLE]
 */
class DMatDistance : public Distance
{
public:

    struct Parameter : public Distance::Parameter
    {
        arma::uword rowSize;

        Parameter(arma::uword size, arma::uword rows) : rowSize(size) 
        {
            total = rows;
        }
    };

public:
    explicit DMatDistance(std::string dmatFile);
    DMatDistance(const DMatDistance& distance);

    virtual Distance * clone() override
    {
        return new DMatDistance(*this);
    }

    DistanceType type() override { return DistanceType::DMatDistance; }

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
    
    virtual arma::vec distance(arma::uword focus) override;
    virtual double maxDistance() override;
    virtual double minDistance() override;

private:
    std::string mDMatFile;
    std::unique_ptr<Parameter> mParameter = nullptr;
};

inline std::string DMatDistance::dMatFile() const
{
    return mDMatFile;
}

inline void DMatDistance::setDMatFile(const std::string &dMatFile)
{
    mDMatFile = dMatFile;
}

}

#endif // DMATDISTANCE_H
