#include "gwmodelpp/spatialweight/CGwmWeight.h"

unordered_map<CGwmWeight::WeightType, string> CGwmWeight::TypeNameMapper = {
    std::make_pair(CGwmWeight::WeightType::BandwidthWeight, "BandwidthWeight")
};
