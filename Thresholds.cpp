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
	ifstream FeatureImageList(FeatureImageFilename.c_str(), std::ofstream::binary);
	vector <FeatureValue> ImageFeature;
	FeatureValue Feature_tmp;
	vector <int> thp, thn;
	int FeatureLen = FeatureLoc.size() * 4;
	for (int fid = 0; fid < FeatureLen; fid++)
	{
		ImageFeature.clear();
		for (int img = 0; img < img_cnt; img++)
		{
			FeatureImageList.read((char*)& Feature_tmp, sizeof(FeatureValue));
			ImageFeature.push_back(Feature_tmp);
		}
		int totalhit = 0;
		int totalsample = 0;
		int save = 0;
		double gini = 0;
		double minGini = 1;
		for (int img = 0; img < img_cnt; img++)
		{
			if (ImageFeature[img].fv > 0)
			{
				totalhit += ImageFeature[img].hit;
				totalsample++;
			}
		}
		for (int img = 0; img < img_cnt; img++)
		{
			double EN = 0;
			double EP = 0;
			if (ImageFeature[img].fv > 0)
			{
				int th = ImageFeature[img].fv;
				int errorNeg = 0;
				int errorPos = 0;
				for (int idx = 0; idx < img_cnt; idx++)
				{
					if (ImageFeature[idx].fv > 0)
					{
						if (ImageFeature[idx].fv < th &&
							ImageFeature[idx].hit == 1)
							errorPos++;
						else if (ImageFeature[idx].fv >= th &&
								 ImageFeature[idx].hit == 0)
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

	for (int fid = 0; fid < FeatureLen; fid++)
	{
		ImageFeature.clear();
		for (int img = 0; img < img_cnt; img++)
		{
			FeatureImageList.read((char*)& Feature_tmp, sizeof(FeatureValue));
			ImageFeature.push_back(Feature_tmp);
		}
		int totalhit = 0;
		int totalsample = 0;
		int save = 0;
		double gini = 0;
		double minGini = 1;		
		for (int img = 0; img < img_cnt; img++)
		{
			if (ImageFeature[img].fv <= 0)
			{
				totalhit += ImageFeature[img].hit;
				totalsample++;
			}
		}

		for (int img = 0; img < img_cnt; img++)
		{
			double EN = 0;
			double EP = 0;
			if (ImageFeature[img].fv <= 0)
			{
				int th = ImageFeature[img].fv;
				int errorNeg = 0;
				int errorPos = 0;
				for (int idx = 0; idx < img_cnt; idx++)
				{
					if (ImageFeature[idx].fv <= 0)
					{
						if (ImageFeature[idx].fv > th &&
							ImageFeature[idx].hit == 1)
							errorPos++;
						else if (ImageFeature[idx].fv <= th &&
							ImageFeature[idx].hit == 0)
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

void BuildThresholdHit(string FeatureImageFilename, vector <int>& ThresholdHit, vector <FeatureThreshold> ThresholdTable, int img_cnt)
{
	ifstream FeatureImageList(FeatureImageFilename.c_str(), std::ifstream::binary);
	FeatureValue Feature_tmp;
	for (int fid = 0; fid < ThresholdTable.size(); fid++)
	{
		for (int img = 0; img < img_cnt; img++)
		{
			FeatureImageList.read((char*)& Feature_tmp, sizeof(FeatureValue));
			if ((Feature_tmp.fv < ThresholdTable[fid].thp && Feature_tmp.fv > ThresholdTable[fid].thn) ||
				(Feature_tmp.fv >= 0 && ThresholdTable[fid].thp == 0) || (Feature_tmp.fv <= 0 && ThresholdTable[fid].thn == 0))
				ThresholdHit.push_back(0);
			else
				ThresholdHit.push_back(1);
		}
	}
	FeatureImageList.close();
}
