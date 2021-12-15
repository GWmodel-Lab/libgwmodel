#include "CGwmGWPCA.h"

CGwmGWPCA::CGwmGWPCA()
{

}

CGwmGWPCA::~CGwmGWPCA()
{
    
}

void CGwmGWPCA::run()
{
    createDistanceParameter();
    setX(mX, mSourceLayer, mVariables);
}