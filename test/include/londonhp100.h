#include <vector>
#include <string>
#include <armadillo>

bool read_londonhp100(arma::mat& coords, arma::mat& data, std::vector<std::string>& fields);

bool read_londonhp100temporal(arma::mat& coords, arma::mat& data, std::vector<std::string>& fields);