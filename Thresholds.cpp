#include "Thresholds.h"
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

void BuildFeatureThreshold(string FeatureImageFilename, vector <TableList> FeatureLoc,
	                       vector <FeatureThreshold>& ThresholdTable, int img_cnt)
{
	ifstream FeatureImageList(FeatureImageFilename.c_str(), std::ifstream::binary);
	FeatureValue* ImageFeature = new FeatureValue[img_cnt * 1000];
//	vector <FeatureValue> ImageFeature(img_cnt);
//	FeatureValue Feature_tmp;
	vector <int> thp, thn;
	int FeatureLen = FeatureLoc.size() * 4;
	cout << "Finding Plus Threshold : ";

	int file_size = 0;
	while (file_size != sizeof(FeatureValue) * FeatureLen * img_cnt)
	{
		FeatureImageList.seekg(0, ios::end);
		file_size = FeatureImageList.tellg();
		FeatureImageList.clear();
		FeatureImageList.seekg(0, ios_base::beg);
	}

	for (int fid = 0; fid < FeatureLen; fid++)
	{
		if (fid % 100000 == 0 || fid == FeatureLen - 1) cout << fid << " ";

		if((fid % 1000 == 0) && ((FeatureLen - fid) / 1000 > 0)) FeatureImageList.read((char*)& ImageFeature[0], sizeof(FeatureValue) * img_cnt * 1000);
		else if((fid % 1000 == 0) && ((FeatureLen - fid) / 1000 == 0)) FeatureImageList.read((char*)& ImageFeature[0], sizeof(FeatureValue) * img_cnt * (FeatureLen % 1000));
	
		int totalhit = 0;
		int totalsample = 0;
		int save = 0;
		double gini = 0;
		double minGini = 1;
		for (int img = 0; img < img_cnt; img++)
		{
			if (ImageFeature[(fid % 1000) * img_cnt + img].fv > 0)
			{
				totalhit += ImageFeature[(fid % 1000) * img_cnt + img].hit;
				totalsample++;
			}
		}
		for (int img = 0; img < img_cnt; img++)
		{
			double EN = 0;
			double EP = 0;
			if (ImageFeature[(fid % 1000) * img_cnt + img].fv > 0)
			{
				int th = ImageFeature[(fid % 1000) * img_cnt + img].fv;
				int errorNeg = 0;
				int errorPos = 0;
				for (int idx = 0; idx < img_cnt; idx++)
				{
					if (ImageFeature[(fid % 1000) * img_cnt + idx].fv > 0)
					{
						if (ImageFeature[(fid % 1000) * img_cnt + idx].fv < th &&
							ImageFeature[(fid % 1000) * img_cnt + idx].hit == 1)
							errorPos++;
						else if (ImageFeature[(fid % 1000) * img_cnt + idx].fv >= th &&
								 ImageFeature[(fid % 1000) * img_cnt + idx].hit == 0)
							errorNeg++;
					}
				}
				if (totalhit == totalsample) EN = 1;
				else EN = ((double)errorNeg) / ((double)(totalsample - totalhit));
				if (totalhit == 0) EP = 1;
				else EP = ((double)errorPos) / ((double) totalhit);
				gini = EN * EN + EP * EP;
				if (gini < minGini)
				{
					save = th;
					minGini = gini;
				}
			}
		}
		thp.push_back(save);
	}

	FeatureImageList.clear();
	FeatureImageList.seekg(0, ios::beg);
	cout << endl << "Finding Minus Threshold : ";

	for (int fid = 0; fid < FeatureLen; fid++)
	{
		if(fid % 100000 == 0 || fid == FeatureLen - 1) cout << fid << " ";

//		FeatureImageList.read((char*)& ImageFeature[0], sizeof(FeatureValue) * img_cnt);
		if ((fid % 1000 == 0) && ((FeatureLen - fid) / 1000 > 0)) FeatureImageList.read((char*)& ImageFeature[0], sizeof(FeatureValue) * img_cnt * 1000);
		else if ((fid % 1000 == 0) && ((FeatureLen - fid) / 1000 == 0)) FeatureImageList.read((char*)& ImageFeature[0], sizeof(FeatureValue) * img_cnt * (FeatureLen % 1000));

		int totalhit = 0;
		int totalsample = 0;
		int save = 0;
		double gini = 0;
		double minGini = 1;		
		for (int img = 0; img < img_cnt; img++)
		{
			if (ImageFeature[(fid % 1000) * img_cnt + img].fv <= 0)
			{
				totalhit += ImageFeature[(fid % 1000) * img_cnt + img].hit;
				totalsample++;
			}
		}

		for (int img = 0; img < img_cnt; img++)
		{
			double EN = 0;
			double EP = 0;
			if (ImageFeature[(fid % 1000) * img_cnt + img].fv <= 0)
			{
				int th = ImageFeature[(fid % 1000) * img_cnt + img].fv;
				int errorNeg = 0;
				int errorPos = 0;
				for (int idx = 0; idx < img_cnt; idx++)
				{
					if (ImageFeature[(fid % 1000) * img_cnt + idx].fv <= 0)
					{
						if (ImageFeature[(fid % 1000) * img_cnt + idx].fv > th &&
							ImageFeature[(fid % 1000) * img_cnt + idx].hit == 1)
							errorPos++;
						else if (ImageFeature[(fid % 1000) * img_cnt + idx].fv <= th &&
							ImageFeature[(fid % 1000) * img_cnt + idx].hit == 0)
							errorNeg++;
					}
				}
				if (totalhit == totalsample) EN = 1;
				else EN = ((double)errorNeg) / ((double)(totalsample - totalhit));
				if (totalhit == 0) EP = 1;
				else EP = ((double)errorPos) / ((double)totalhit);
				gini = EN * EN + EP * EP;
				if (gini < minGini)
				{
					save = th;
					minGini = gini;
				}
			}
		}
		thn.push_back(save);
	}
	FeatureImageList.close();
	cout << endl;

	FeatureThreshold vtmp;
	for (int fid = 0; fid < FeatureLen; fid++)
	{
		vtmp.box = FeatureLoc[fid / 4];
		vtmp.box.id = fid;
		vtmp.thp = thp[fid];
		vtmp.thn = thn[fid];
		ThresholdTable.push_back(vtmp);
	}
}

void BuildThresholdHit(string FeatureImageFilename, string ThresholdHitFilename, vector <FeatureThreshold> ThresholdTable, int img_cnt)
{
	cout << endl << "BuildThresholdHit......" << endl;

	ifstream FeatureImageList(FeatureImageFilename.c_str(), std::ifstream::binary);
	ofstream ThresholdHit(ThresholdHitFilename.c_str(), std::ofstream::binary);
	FeatureValue* Feature_tmp = new FeatureValue[img_cnt * 1000];
	int* Hit_tmp = new int[img_cnt * 1000];
	int FeatureLen = ThresholdTable.size();
	int tt=0;

	for (int fid = 0; fid < FeatureLen; fid++)
	{
		if ((fid % 1000 == 0) && ((FeatureLen - fid) / 1000 > 0)) FeatureImageList.read((char*)& Feature_tmp[0], sizeof(FeatureValue) * img_cnt * 1000);
		else if ((fid % 1000 == 0) && ((FeatureLen - fid) / 1000 == 0)) FeatureImageList.read((char*)& Feature_tmp[0], sizeof(FeatureValue) * img_cnt * (FeatureLen % 1000));

		for (int img = 0; img < img_cnt; img++)
		{
			if ((Feature_tmp[(fid % 1000) * img_cnt + img].fv < ThresholdTable[fid].thp && Feature_tmp[(fid % 1000) * img_cnt + img].fv > ThresholdTable[fid].thn) ||
				(Feature_tmp[(fid % 1000) * img_cnt + img].fv >= 0 && ThresholdTable[fid].thp == 0) || (Feature_tmp[(fid % 1000) * img_cnt + img].fv <= 0 && ThresholdTable[fid].thn == 0))
				Hit_tmp[(fid % 1000) * img_cnt + img] = 0;
			else
				Hit_tmp[(fid % 1000) * img_cnt + img] = 1;
		}
		if (((fid + 1) % 1000 == 0) && ((fid + 1) / 1000 > 0))
		{
			ThresholdHit.write((char*)& Hit_tmp[0], sizeof(int) * img_cnt * 1000);
			tt++;
		}
	}
	if (FeatureLen % 1000 > 0)
	{
		ThresholdHit.write((char*)& Hit_tmp[0], sizeof(int) * img_cnt * (FeatureLen % 1000));
		tt++;
	}

	FeatureImageList.close();
	ThresholdHit.close();
}
