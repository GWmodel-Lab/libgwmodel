#include "CGwmGWPCA.h"

CGwmGWPCA::CGwmGWPCA()
{

}

CGwmGWPCA::~CGwmGWPCA()
{
    
}

CGwmGWPCA::run()
{
    createDistanceParameter();
    setX(mX, mSourceLayer, mVariables);
}