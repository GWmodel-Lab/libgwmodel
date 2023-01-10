#include "gwmodelpp/spatialweight/CGwmDistance.h"
#include <assert.h>

using namespace std;

unordered_map<CGwmDistance::DistanceType, string> CGwmDistance::TypeNameMapper =
{
    std::make_pair(CGwmDistance::DistanceType::CRSDistance, "CRSDistance"),
    std::make_pair(CGwmDistance::DistanceType::MinkwoskiDistance, "MinkwoskiDistance"),
    std::make_pair(CGwmDistance::DistanceType::DMatDistance, "DMatDistance")
};
