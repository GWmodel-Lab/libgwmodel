#include "CGwmGWRBase.h"
#include <assert.h>

bool CGwmGWRBase::isValid()
{
    if (CGwmSpatialMonoscaleAlgorithm::isValid())
    {
        if (!(mX.n_cols > 0))
            return false;
        
        if (!(mX.n_rows == mY.n_rows && mX.n_rows == mCoords.n_rows)) 
            return false;

        return true;
    }
    else return false;
}