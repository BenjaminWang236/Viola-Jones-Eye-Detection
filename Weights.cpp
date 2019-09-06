#include "Weights.h"
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

void initWeights(string FeatureImageFilename, double** weights,
	             vector <FeatureThreshold> ThresholdTable, int img_cnt)
{
	ifstream FeatureImageList(FeatureImageFilename.c_str(), std::ifstream::binary);
	FeatureValue Feature_tmp;
	vector <FeatureValue> ImageFeature;
	for (int fid = 0; fid < ThresholdTable.size(); fid++)
	{
		ImageFeature.clear();
		for (int img = 0; img < img_cnt; img++)
		{
			FeatureImageList.read((char*)& Feature_tmp, sizeof(FeatureValue));
			ImageFeature.push_back(Feature_tmp);
		}
		int negcnt = 0;
		int poscnt = 0;
		for (int img = 0; img < img_cnt; img++)
		{
			if (ImageFeature[img].hit == 0) negcnt++;
			else poscnt++;
		}
		for (int img = 0; img < img_cnt; img++)
		{
			if (ImageFeature[img].hit == 0)
			{
				if (poscnt == 0) weights[fid][img] = 1.0 / double(negcnt);
				else weights[fid][img] = 1.0 / double(negcnt * 2);
			}
			else
			{
				if (negcnt == 0) weights[fid][img] = 1.0 / double(poscnt);
				else weights[fid][img] = 1.0 / double(poscnt * 2);
			}
		}
	}
	FeatureImageList.close();
}

void normWeights(double** weights, vector <FeatureThreshold> ThresholdTable, int img_cnt)
{
	double weightSum = 0;
	for (int f = 0; f < ThresholdTable.size(); f++)
	{
		for (int i = 0; i < img_cnt; i++) weightSum += weights[f][i];
	}
	if (weightSum == 0) weightSum = 1.0;
	for (int f = 0; f < ThresholdTable.size(); f++)
	{
		for (int i = 0; i < img_cnt; i++) weights[f][i] /= weightSum;
	}
}

void updateWeights(double** weights, vector <int> ThresholdHit, string FeatureImageFilename, vector <FeatureThreshold> ThresholdTable, int img_cnt, int minIndex, double beta)
{
	ifstream FeatureImageList(FeatureImageFilename.c_str(), std::ifstream::binary);
	FeatureValue Feature_tmp;
	int FeatureLen = ThresholdTable.size();

	FeatureImageList.seekg(minIndex * img_cnt * sizeof(FeatureValue), ios::beg);

	for (int img = 0; img < img_cnt; img++)
	{
		FeatureImageList.read((char*)& Feature_tmp, sizeof(FeatureValue));
		if (Feature_tmp.hit == ThresholdHit[minIndex * img_cnt + img]) weights[minIndex][img] *= beta;
	}
	FeatureImageList.close();
}
