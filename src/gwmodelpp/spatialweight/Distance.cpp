#include "gwmodelpp/spatialweight/Distance.h"
#include <assert.h>

using namespace std;
using namespace gwm;

unordered_map<Distance::DistanceType, string> Distance::TypeNameMapper =
{
    std::make_pair(Distance::DistanceType::CRSDistance, "CRSDistance"),
    std::make_pair(Distance::DistanceType::MinkwoskiDistance, "MinkwoskiDistance"),
    std::make_pair(Distance::DistanceType::DMatDistance, "DMatDistance")
};
