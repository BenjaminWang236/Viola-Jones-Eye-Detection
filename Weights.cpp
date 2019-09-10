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

void initWeights(string FeatureImageFilename, string Weight0Filename, int FeatureLen, int img_cnt)
{
	cout << endl << "initWeights......" << endl;

	ifstream FeatureImageList(FeatureImageFilename.c_str(), std::ifstream::binary);
	ofstream WeightInit(Weight0Filename.c_str(), std::ofstream::binary);
	FeatureValue* ImageFeature = new FeatureValue[img_cnt * 1000];
	double* Weights = new double[img_cnt * 1000];

	for (int fid = 0; fid < FeatureLen; fid++)
	{
		if ((fid % 1000 == 0) && ((FeatureLen - fid) / 1000 > 0)) FeatureImageList.read((char*)& ImageFeature[0], sizeof(FeatureValue) * img_cnt * 1000);
		else if ((fid % 1000 == 0) && ((FeatureLen - fid) / 1000 == 0)) FeatureImageList.read((char*)& ImageFeature[0], sizeof(FeatureValue) * img_cnt * (FeatureLen % 1000));
		int negcnt = 0;
		int poscnt = 0;
		for (int img = 0; img < img_cnt; img++)
		{
			if (ImageFeature[(fid % 1000) * img_cnt + img].hit == 0) negcnt++;
			else poscnt++;
		}
		for (int img = 0; img < img_cnt; img++)
		{
			if (ImageFeature[(fid % 1000) * img_cnt + img].hit == 0)
			{
				if (poscnt == 0) Weights[(fid % 1000) * img_cnt + img] = (1.0 / double(negcnt));
				else Weights[(fid % 1000) * img_cnt + img] = (1.0 / double(negcnt * 2));
			}
			else
			{
				if (negcnt == 0) Weights[(fid % 1000) * img_cnt + img] = (1.0 / double(poscnt));
				else Weights[(fid % 1000) * img_cnt + img] = (1.0 / double(poscnt * 2));
			}
		}
		if (((fid + 1) % 1000 == 0) && ((fid + 1) / 1000 > 0))
		{
			WeightInit.write((char*)& Weights[0], sizeof(double) * img_cnt * 1000);
			for (int i = 0; i < 1000;i++)
			{
				double ttt = Weights[i];
			}
		}
	}
	WeightInit.write((char*)& Weights[0], sizeof(double) * img_cnt * (FeatureLen % 1000));
	for (int i = 0; i < (FeatureLen % 1000);i++)
	{
		double ttt = Weights[i];
	}
	FeatureImageList.close();
	WeightInit.close();
}

void normWeights(string Weight0Filename, string WeightNormalFilename, int FeatureLen, int img_cnt)
{
	
	cout << endl << "normWeights......" << endl;
	ifstream Weight0(Weight0Filename.c_str(), std::ifstream::binary);
	ofstream WeightNormal(WeightNormalFilename.c_str(), std::ofstream::binary);

	double WeightSum = 0;
	double* Weights_tmp = new double[img_cnt * 1000];
	
	int file_size = 0;
	while (file_size != sizeof(double) * FeatureLen * img_cnt)
	{
		Weight0.seekg(0, ios_base::end);
		file_size = Weight0.tellg();
		Weight0.clear();
		Weight0.seekg(0, ios_base::beg);
	}
	
	for (int fid = 0; fid < FeatureLen; fid++)
	{
		if ((fid % 1000 == 0) && ((FeatureLen - fid) / 1000 > 0)) Weight0.read((char*)& Weights_tmp[0], sizeof(double) * img_cnt * 1000);
		else if ((fid % 1000 == 0) && ((FeatureLen - fid) / 1000 == 0)) Weight0.read((char*)& Weights_tmp[0], sizeof(double) * img_cnt * (FeatureLen % 1000));

//		Weight0.read((char*)& Weights_tmp[0], sizeof(double) * img_cnt);
		for (int i = 0; i < img_cnt; i++) WeightSum += Weights_tmp[(fid % 1000)* img_cnt + i];
	}
	if (WeightSum == 0) WeightSum = 1.0;

	Weight0.seekg(0, ios::beg);
	for (int fid = 0; fid < FeatureLen; fid++)
	{
		if ((fid % 1000 == 0) && ((FeatureLen - fid) / 1000 > 0)) Weight0.read((char*)& Weights_tmp[0], sizeof(double) * img_cnt * 1000);
		else if ((fid % 1000 == 0) && ((FeatureLen - fid) / 1000 == 0)) Weight0.read((char*)& Weights_tmp[0], sizeof(double) * img_cnt * (FeatureLen % 1000));
		//		Weight0.read((char*)& Weights_tmp[0], sizeof(double) * img_cnt);
		for (int i = 0; i < img_cnt; i++) Weights_tmp[(fid % 1000)* img_cnt + i] /= WeightSum;
//		WeightNormal.write((char*)& Weights_tmp[0], sizeof(double) * img_cnt);
		if (((fid + 1) % 1000 == 0) && ((fid + 1) / 1000 > 0)) WeightNormal.write((char*)& Weights_tmp[0], sizeof(double) * img_cnt * 1000);
	}
	WeightNormal.write((char*)& Weights_tmp[0], sizeof(double) * img_cnt * (FeatureLen % 1000));

	WeightNormal.close();
	Weight0.close();
}

void updateWeights(string Weight0Filename, string WeightNormalFilename, string ThresholdHitFilename, string FeatureImageFilename, int FeatureLen, int img_cnt, int minIndex, double beta)
{
	cout << endl << "updateWeights......" << endl;

	ifstream FeatureImageList(FeatureImageFilename.c_str(), std::ifstream::binary);
	ifstream WeightNormal(Weight0Filename.c_str(), std::ifstream::binary);
	ofstream Weight0(WeightNormalFilename.c_str(), std::ofstream::binary);
	ifstream ThresholdHit(ThresholdHitFilename.c_str(), std::ifstream::binary);
	FeatureValue *Feature_tmp = new FeatureValue[img_cnt];
	double* Weights_tmp = new double[img_cnt * 1000];
	int* Hit_tmp = new int[img_cnt];

	for (int fid = 0; fid < ((minIndex - 1) / 1000) * 1000; fid = fid + 1000)
	{
		WeightNormal.read((char*)& Weights_tmp[0], sizeof(double) * img_cnt * 1000);
		Weight0.write((char*)& Weights_tmp[0], sizeof(double) * img_cnt * 1000);
	}
	if (minIndex > 0)
	{
		WeightNormal.read((char*)& Weights_tmp[0], sizeof(double) * img_cnt * ((minIndex - 1) % 1000));
		Weight0.write((char*)& Weights_tmp[0], sizeof(double) * img_cnt * ((minIndex - 1) % 1000));
	}

	FeatureImageList.seekg(minIndex * img_cnt * sizeof(FeatureValue), ios::beg);
	FeatureImageList.read((char*)& Feature_tmp[0], sizeof(FeatureValue) * img_cnt);
	WeightNormal.read((char*)& Weights_tmp[0], sizeof(double) * img_cnt);
	ThresholdHit.read((char*)& Hit_tmp[0], sizeof(int) * img_cnt);

	for (int img = 0; img < img_cnt; img++)
	{
		if (Feature_tmp[img].hit == Hit_tmp[img]) Weights_tmp[img] *= beta;
	}

	Weight0.write((char*)& Weights_tmp[0], sizeof(double) * img_cnt);


	WeightNormal.read((char*)& Weights_tmp[0], sizeof(double) * img_cnt * ((FeatureLen - (minIndex + 1)) % 1000));
	Weight0.write((char*)& Weights_tmp[0], sizeof(double) * img_cnt * ((FeatureLen - (minIndex + 1)) % 1000));

	for (int fid = (FeatureLen - (minIndex + 1))/1000; fid == 0; fid--)
	{
		WeightNormal.read((char*)& Weights_tmp[0], sizeof(double) * img_cnt * 1000);
		Weight0.write((char*)& Weights_tmp[0], sizeof(double) * img_cnt * 1000);
	}
	FeatureImageList.close();
	WeightNormal.close();
	Weight0.close();
	ThresholdHit.close();
}
