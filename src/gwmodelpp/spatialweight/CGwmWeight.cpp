#include "gwmodelpp/spatialweight/CGwmWeight.h"

using namespace std;
using namespace gwm;

unordered_map<CGwmWeight::WeightType, string> CGwmWeight::TypeNameMapper = {
    std::make_pair(CGwmWeight::WeightType::BandwidthWeight, "BandwidthWeight")
};
