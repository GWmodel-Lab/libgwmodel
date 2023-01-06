/**
 * @file gwmodel.h
 * @author YigongHu (hu_yigong@whu.edu.cn)
 * @brief This file provide the headers of this library. 
 * If the library is built as a static library, this header includes all the C++ headers. 
 * If the library is built as a shared library, this header provides interface functions for the calling of C++ classes.
 * @version 0.1.0
 * @date 2020-10-08
 * 
 * @copyright Copyright (c) 2020
 * 
 * 
 */

#ifndef GWMODEL_H
#define GWMODEL_H

#include "gwmodelpp/spatialweight/CGwmBandwidthWeight.h"
#include "gwmodelpp/spatialweight/CGwmCRSDistance.h"
#include "gwmodelpp/spatialweight/CGwmDistance.h"
#include "gwmodelpp/spatialweight/CGwmDMatDistance.h"
#include "gwmodelpp/spatialweight/CGwmMinkwoskiDistance.h"
#include "gwmodelpp/spatialweight/CGwmSpatialWeight.h"
#include "gwmodelpp/spatialweight/CGwmWeight.h"

#include "gwmodelpp/CGwmAlgorithm.h"
#include "gwmodelpp/CGwmBandwidthSelector.h"
#include "gwmodelpp/CGwmVariableForwardSelector.h"
#include "gwmodelpp/CGwmSpatialAlgorithm.h"
#include "gwmodelpp/CGwmSpatialMonoscaleAlgorithm.h"
#include "gwmodelpp/CGwmGWRBase.h"
#include "gwmodelpp/CGwmGWRBasic.h"
#include "gwmodelpp/CGwmGWPCA.h"
#include "gwmodelpp/CGwmMGWR.h"

#endif  // GWMODEL_H