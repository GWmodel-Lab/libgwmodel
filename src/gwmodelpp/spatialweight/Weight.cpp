#include "gwmodelpp/spatialweight/Weight.h"

using namespace std;
using namespace gwm;

unordered_map<Weight::WeightType, string> Weight::TypeNameMapper = {
    std::make_pair(Weight::WeightType::BandwidthWeight, "BandwidthWeight")
};
