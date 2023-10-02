#ifndef GWRBASICGPUTASK
#define GWRBASICGPUTASK

#include <memory>
#include <armadillo>

#include "IGWRBasicGpuTask.h"
#include "../gwmodel.h"

class GWRBasicGpuTask : public IGWRBasicGpuTask
{
private:
	arma::mat mX;
	arma::vec mY;
	arma::mat mCoords;
	arma::mat mPredictLocations;
	arma::mat mBetas;
	arma::mat mBetasSE;
	arma::vec mSHat;
	arma::vec mQDiag;

	gwm::Distance* mDistance = nullptr;
	gwm::Weight* mWeight = nullptr;

	bool mIsOptimizeBandwidth = false;
	gwm::GWRBasic::BandwidthSelectionCriterionType mBandwidthOptimizationCriterion = gwm::GWRBasic::BandwidthSelectionCriterionType::CV;

	bool mIsOptimizeVariables = false;

public:
	GWRBasicGpuTask(int nDp, int nVar, gwm::Distance::DistanceType distanceType) :
		mX(nDp, nVar),
		mY(nDp),
		mCoords(nDp, 2),
		mBetas(nDp, nVar),
		mBetasSE(nDp, nVar),
		mSHat(2),
		mQDiag(nDp)
	{
		switch (distanceType)
		{
		case gwm::Distance::DistanceType::CRSDistance:
			mDistance = new gwm::CRSDistance();
			break;
		case gwm::Distance::DistanceType::MinkwoskiDistance:
			mDistance = new gwm::MinkwoskiDistance();
		default:
			break;
		}
		mWeight = new gwm::BandwidthWeight();
	}

	GWRBasicGpuTask(int nDp, int nVar, gwm::Distance::DistanceType distanceType, int nPredictPoints) :
		mX(nDp, nVar),
		mY(nDp),
		mCoords(nDp, 2),
		mBetas(nPredictPoints, nVar),
		mPredictLocations(nPredictPoints, 2)
	{
	}

	GWRBasicGpuTask(const GWRBasicGpuTask& source) :
		mX(source.mX),
		mY(source.mY),
		mCoords(source.mCoords),
		mPredictLocations(source.mPredictLocations),
		mBetas(source.mBetas),
		mBetasSE(source.mBetasSE),
		mSHat(source.mSHat),
		mQDiag(source.mQDiag),
		mIsOptimizeBandwidth(source.mIsOptimizeBandwidth),
		mBandwidthOptimizationCriterion(source.mBandwidthOptimizationCriterion),
		mIsOptimizeVariables(source.mIsOptimizeVariables)
	{
		mDistance = source.mDistance->clone();
		mWeight = source.mWeight->clone();
	}

	~GWRBasicGpuTask()
	{
		if (mDistance) delete mDistance;
		if (mWeight) delete mWeight;
	}

	GWRBasicGpuTask& operator=(const GWRBasicGpuTask& source)
	{
		mX = source.mX;
		mY = source.mY;
		mCoords = source.mCoords;
		mPredictLocations = source.mPredictLocations;
		mBetas = source.mBetas;
		mBetasSE = source.mBetasSE;
		mSHat = source.mSHat;
		mQDiag = source.mQDiag;
		mDistance = source.mDistance->clone();
		mWeight = source.mWeight->clone();
		mIsOptimizeBandwidth = source.mIsOptimizeBandwidth;
		mBandwidthOptimizationCriterion = source.mBandwidthOptimizationCriterion;
		mIsOptimizeVariables = source.mIsOptimizeVariables;
		return *this;
	}

	void setX(int i, int k, double value) override
	{
		mX(i, k) = value;
	}

	void setY(int i, double value) override
	{
		mY(i) = value;
	}

	void setCoords(int i, double u, double v) override
	{
		mCoords(i, u) = v;
	}

	void setPredictLocations(int i, double u, double v) override
	{
		mPredictLocations(i, u) = v;
	}

	void setDistanceType(int type) override
	{
		switch ((gwm::Distance::DistanceType)type)
		{
		case gwm::Distance::DistanceType::CRSDistance:
			mDistance = new gwm::CRSDistance();
			break;
		case gwm::Distance::DistanceType::MinkwoskiDistance:
			mDistance = new gwm::MinkwoskiDistance();
		default:
			break;
		}
	}

	void setCRSDistanceGergraphic(bool isGeographic) override
	{
		static_cast<gwm::CRSDistance*>(mDistance)->setGeographic(isGeographic);
	}

	void setMinkwoskiDistancePoly(int poly) override
	{
		static_cast<gwm::MinkwoskiDistance*>(mDistance)->setPoly(poly);
	}

	void setMinkwoskiDistanceTheta(double theta) override
	{
		static_cast<gwm::MinkwoskiDistance*>(mDistance)->setTheta(theta);
	}


	void setBandwidthSize(double bw) override
	{
		static_cast<gwm::BandwidthWeight*>(mWeight)->setBandwidth(bw);
	}

	void setBandwidthAdaptive(bool adaptive) override
	{
		static_cast<gwm::BandwidthWeight*>(mWeight)->setAdaptive(adaptive);
	}

	void setBandwidthKernel(int kernel) override
	{
		static_cast<gwm::BandwidthWeight*>(mWeight)->setKernel((gwm::BandwidthWeight::KernelFunctionType)kernel);
	}

	void enableBandwidthOptimization(int criterion) override
	{
		mIsOptimizeBandwidth = true;
		mBandwidthOptimizationCriterion = static_cast<gwm::GWRBasic::BandwidthSelectionCriterionType>(criterion);
	}

	void enableVariablesOptimization() override
	{
		mIsOptimizeVariables = true;
	}

	double betas(int i, int k) override
	{
		return betas(i, k);
	}

	double betasSE(int i, int k) override
	{
		return betasSE(i, k);
	}

	double shat1() override
	{
		return mSHat(0);
	}

	double shat2() override
	{
		return mSHat(1);
	}

	double qDiag(int i) override
	{
		return mQDiag(i);
	}

	bool fit(bool hasIntercept = true) override;

	bool predict(bool hasIntercept) override;

};

#endif  // GWRBASICGPUTASK
