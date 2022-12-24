#include "londonhp.h"

bool read_londonhp(mat& coords, mat& data, vector<string>& fields)
{
    field<std::string> coordHeader(2);
    coordHeader(0) = "x";
    coordHeader(1) = "y";
    bool coords_loaded = coords.load(arma::csv_name(string(SAMPLE_DATA_DIR) + "/londonhpcoords.csv", coordHeader));

    field<std::string> dataHeader(20);
    dataHeader(0) = "PURCHASE";
    dataHeader(1) = "FLOORSZ";
    dataHeader(2) = "TYPEDETCH";
    dataHeader(3) = "TPSEMIDTCH";
    dataHeader(4) = "TYPETRRD";
    dataHeader(5) = "TYPEBNGLW";
    dataHeader(6) = "TYPEFLAT";
    dataHeader(7) = "BLDPWW1";
    dataHeader(8) = "BLDPOSTW";
    dataHeader(9) = "BLD60S";
    dataHeader(10) = "BLD70S";
    dataHeader(11) = "BLD80S";
    dataHeader(12) = "BLD90S";
    dataHeader(13) = "BATH2";
    dataHeader(14) = "BEDS2";
    dataHeader(15) = "GARAGE1";
    dataHeader(16) = "CENTHEAT";
    dataHeader(17) = "UNEMPLOY";
    dataHeader(18) = "PROF";
    dataHeader(19) = "BLDINTW";
    bool data_loaded = data.load(arma::csv_name(string(SAMPLE_DATA_DIR) + "/londonhpdata.csv", dataHeader));

    if (coords_loaded && data_loaded)
    {
        fields = { "PURCHASE", "FLOORSZ", "TYPEDETCH", "TPSEMIDTCH", "TYPETRRD","TYPEBNGLW", "TYPEFLAT", "BLDPWW1",
        "BLDPOSTW","BLD60S","BLD70S","BLD80S","BLD90S","BATH2","BEDS2","GARAGE1","CENTHEAT","UNEMPLOY","PROF","BLDINTW" };
        return true;
    }
    else return false;
}