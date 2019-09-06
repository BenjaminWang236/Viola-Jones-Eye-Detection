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
#define DEBUG_FEATURE

TrainOut train(std::ofstream& TableOut, double** weights, vector <int> ThresholdHit,
	string FeatureImageFilename, vector <FeatureThreshold> ThresholdTable, vector <int> MinIndex, int img_cnt)
{
	ifstream FeatureImageList(FeatureImageFilename.c_str(), std::ifstream::binary);
	FeatureValue Feature_tmp;
	vector <double> featureError;
	double alpha;
	int FeatureLen = ThresholdTable.size();
	for (int fid = 0; fid < FeatureLen; fid++)
	{
		double sum = 0;
		for (int img = 0; img < img_cnt; img++)
		{
			FeatureImageList.read((char*)& Feature_tmp, sizeof(FeatureValue));
			if (ThresholdHit[fid * img_cnt + img] != Feature_tmp.hit)
			{
				sum += weights[fid][img];
			}
		}
		for (int i = 0; i < MinIndex.size(); i++)
		{
			if (fid == MinIndex[i]) sum = 100.0;
		}
		featureError.push_back(sum);
	}
	FeatureImageList.close();

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
			<< "	" << alpha
			<< "    " << endl;
	}

/*
	TableOut << setw(5) << index << setw(4) << index % 4
		<< setw(6) << ThresholdTable[index].box.xs
		<< setw(6) << ThresholdTable[index].box.ys
		<< setw(4) << ThresholdTable[index].box.xe
		<< setw(4) << ThresholdTable[index].box.ye
		<< setw(12) << ThresholdTable[index].thp
		<< setw(12) << ThresholdTable[index].thn
		<< setw(12) << alpha;
*/
/*
	fprintf(TableOut, "%5d %4d %6d %6d %4d %4d %12d %12d %12f\n",
		index, index % 4, ThresholdTable[index].box.xs, ThresholdTable[index].box.ys,
		ThresholdTable[index].box.xe, ThresholdTable[index].box.ye,
		ThresholdTable[index].thp, ThresholdTable[index].thn, alpha);
		*/
	TrainOut tmp;
	tmp.beta = beta;
	tmp.minidx = index;
	return tmp;
}
