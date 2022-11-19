#include "londonhp100.h"

using namespace std;
using namespace arma;

bool read_londonhp100(mat& coords, mat& data, vector<string>& fields)
{
    field<std::string> coordHeader(2);
    coordHeader(0) = "x";
    coordHeader(1) = "y";
    bool coords_loaded = coords.load(arma::csv_name(string(SAMPLE_DATA_DIR) + "/londonhp100coords.csv", coordHeader));

    field<std::string> dataHeader(4);
    dataHeader(0) = "PURCHASE";
    dataHeader(1) = "FLOORSZ";
    dataHeader(2) = "UNEMPLOY";
    dataHeader(3) = "PROF";
    bool data_loaded = data.load(arma::csv_name(string(SAMPLE_DATA_DIR) + "/londonhp100data.csv", dataHeader));

    if (coords_loaded && data_loaded)
    {
        fields = { "PURCHASE", "FLOORSZ", "UNEMPLOY", "PROF" };
        return true;
    }
    else return false;
}

bool read_londonhp100temporal(mat& coords, mat& data, vector<string>& fields)
{
    field<std::string> coordHeader(3);
    coordHeader(0) = "x";
    coordHeader(1) = "y";
    coordHeader(2) = "t";
    bool coords_loaded = coords.load(arma::csv_name(string(SAMPLE_DATA_DIR) + "/londonhp100temcoords.csv", coordHeader));

    field<std::string> dataHeader(4);
    dataHeader(0) = "PURCHASE";
    dataHeader(1) = "FLOORSZ";
    dataHeader(2) = "UNEMPLOY";
    dataHeader(3) = "PROF";
    bool data_loaded = data.load(arma::csv_name(string(SAMPLE_DATA_DIR) + "/londonhp100data.csv", dataHeader));

    if (coords_loaded && data_loaded)
    {
        fields = { "PURCHASE", "FLOORSZ", "UNEMPLOY", "PROF" };
        return true;
    }
    else return false;
}