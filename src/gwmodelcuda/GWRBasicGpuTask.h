#ifndef GWRBASICGPUTASK
#define GWRBASICGPUTASK

#include <armadillo>

#include "IGpuTask.h"

class GWRBasicGpuTask : public IGWRBasicGpuTask
{
private:
	arma::mat x;
	arma::vec y;
	arma::mat dp;
	arma::mat rp;
	arma::mat dMat;
	bool rp_given;
	bool dm_given;
	arma::mat betas;
	arma::mat betasSE;
	arma::vec s_hat;
	arma::vec qdiag;

public:
	GWRBasicGpuTask();
	GWRBasicGpuTask(int N, int K, bool rp_given, int n, bool dm_given);
	~GWRBasicGpuTask();

	virtual void setX(int i, int k, double value) override;
	virtual void setY(int i, double value) override;
	virtual void setDp(int i, double u, double v) override;
	virtual void setRp(int i, double u, double v) override;
	virtual void setDmat(int i, int j, double value) override;

	virtual double betas(int i, int k) override;
	virtual double betasSE(int i, int k) override;
	virtual double shat1() override;
	virtual double shat2() override;
	virtual double qdiag(int i) override;


	virtual bool fit(
		bool hatmatrix,
		double p, double theta, bool longlat,
		double bw, int kernel, bool adaptive,
		int groupl, int gpuID
	) override;

	virtual bool predict(
		double p, double theta, bool longlat,
		double bw, int kernel, bool adaptive,
		int groupl, int gpuID
	) override;

	virtual double cv(
		double p, double theta, bool longlat,
		double bw, int kernel, bool adaptive,
		int groupl, int gpuID
	) override;

};

#endif  // GWRBASICGPUTASK
