#include "Train.h"
#include <iostream>
#include <fstream>
#include <string>
#include <cstdlib>
#include <vector>
#include <sstream>
#include <cmath>
#include <algorithm>
#include <iomanip>      // std::setw
#include <chrono>
#include "features.h"

using namespace std;
using namespace std::chrono;
#define DEBUG_FEATURE

string format_duration(long dur) {
	microseconds milsec(dur);
	auto ms = duration_cast<milliseconds>(milsec);		milsec -= duration_cast<microseconds>(ms);
	auto secs = duration_cast<seconds>(ms);				ms -= duration_cast<milliseconds>(secs);
	auto mins = duration_cast<minutes>(secs);			secs -= duration_cast<seconds>(mins);
	auto hour = duration_cast<hours>(mins);				mins -= duration_cast<minutes>(hour);

	stringstream ss;
	ss << hour.count() << " hours " << mins.count() << " minutes " << secs.count() << " seconds " <<
			ms.count() << " milliseconds " << milsec.count() << " microseconds";
	return ss.str();
}

TrainOut train(std::ofstream& TableOut, string WeightNormalFilename, string ThresholdHitFilename,
	string FeatureImageFilename, vector <FeatureThreshold> ThresholdTable, vector <int> MinIndex, int img_cnt)
{
	cout << endl << "train......" << endl;

	ifstream FeatureImageList(FeatureImageFilename.c_str(), std::ifstream::binary);
	ifstream WeightNormal(WeightNormalFilename.c_str(), std::ifstream::binary);
	ifstream ThresholdHit(ThresholdHitFilename.c_str(), std::ifstream::binary);

	FeatureValue* Feature_tmp = new FeatureValue[img_cnt * 1000];
	vector <double> featureError;
	double alpha;
	double* Weights_tmp = new double[img_cnt * 1000];
	int* Hit_tmp = new int[img_cnt * 1000];
	int FeatureLen = ThresholdTable.size();

	int file_size = 0;

	while (file_size != sizeof(int) * FeatureLen * img_cnt)
	{
		ThresholdHit.seekg(0, ios_base::end);
		file_size = ThresholdHit.tellg();
		ThresholdHit.clear();
		ThresholdHit.seekg(0, ios_base::beg);
	}

	while (file_size != sizeof(double) * FeatureLen * img_cnt)
	{
		WeightNormal.seekg(0, ios_base::end);
		file_size = WeightNormal.tellg();
		WeightNormal.clear();
		WeightNormal.seekg(0, ios_base::beg);
	}

	for (int fid = 0; fid < FeatureLen; fid++)
	{
		double sum = 0;
		if ((fid % 1000 == 0) && ((FeatureLen - fid) / 1000 > 0)) WeightNormal.read((char*)& Weights_tmp[0], sizeof(double) * img_cnt * 1000);
		else if ((fid % 1000 == 0) && ((FeatureLen - fid) / 1000 == 0)) WeightNormal.read((char*)& Weights_tmp[0], sizeof(double) * img_cnt * (FeatureLen % 1000));

		if ((fid % 1000 == 0) && ((FeatureLen - fid) / 1000 > 0)) FeatureImageList.read((char*)& Feature_tmp[0], sizeof(FeatureValue) * img_cnt * 1000);
		else if ((fid % 1000 == 0) && ((FeatureLen - fid) / 1000 == 0)) FeatureImageList.read((char*)& Feature_tmp[0], sizeof(FeatureValue) * img_cnt * (FeatureLen % 1000));

		if ((fid % 1000 == 0) && ((FeatureLen - fid) / 1000 > 0)) ThresholdHit.read((char*)& Hit_tmp[0], sizeof(int) * img_cnt * 1000);
		else if ((fid % 1000 == 0) && ((FeatureLen - fid) / 1000 == 0)) ThresholdHit.read((char*)& Hit_tmp[0], sizeof(int) * img_cnt * (FeatureLen % 1000));

		for (int img = 0; img < img_cnt; img++)
		{
			if (Hit_tmp[(fid % 1000) * img_cnt + img] != Feature_tmp[(fid % 1000) * img_cnt + img].hit) sum += Weights_tmp[(fid % 1000) * img_cnt + img];
		}
		vector <int>::iterator it = find(MinIndex.begin(), MinIndex.end(), fid);
		if (it != MinIndex.end())
		{
			sum = 100.0;
		}

		featureError.push_back(sum);
	}
	FeatureImageList.close();
	WeightNormal.close();
	ThresholdHit.close();

	int index = std::min_element(featureError.begin(), featureError.end()) - featureError.begin();
	double minError = *std::min_element(featureError.begin(), featureError.end());

	double beta = minError / (1.0 - minError);
	if (beta == 0) alpha = 50;
	else alpha = log(1 / beta);

	if (TableOut.is_open() )
	{
		TableOut << index << "	" << index % 4
			<< "	" << ThresholdTable[index].box.xs
			<< "	" << ThresholdTable[index].box.ys
			<< "	" << ThresholdTable[index].box.xe
			<< "	" << ThresholdTable[index].box.ye
			<< "	" << ThresholdTable[index].thp
			<< "	" << ThresholdTable[index].thn
			<< "	" << alpha << endl;
	}

	TrainOut tmp;
	tmp.beta = beta;
	tmp.minidx = index;
	return tmp;
}
