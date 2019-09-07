#include "features.h"
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

using namespace std;
#define DEBUG_FEATURE

void BuildFeatureLoc(vector <TableList>& FeatureLoc, vector <TableList> LeftMinMax, vector <TableList> RightMinMax, int FaceWidth)
{
	TableList Table_tmp;
	bool same = false;
	int id = 0;

	int xblock = 2, yblock = 2, offset = FaceWidth/16;
	int ysmin = LeftMinMax[0].ys - offset; if (ysmin < 0) ysmin = 0;
	int xsmin = LeftMinMax[0].xs - offset; if (xsmin < 0) xsmin = 0;
	int yemin = LeftMinMax[0].ye - offset; if (yemin < 0) yemin = 0;
	int xemin = LeftMinMax[0].xe - offset; if (xemin < 0) xemin = 0;	
	int ysmax = LeftMinMax[1].ys + offset; if (ysmax > 31) ysmax = 31;
	int xsmax = LeftMinMax[1].xs + offset; if (xsmax > 31) xsmax = 31;
	int yemax = LeftMinMax[1].ye + offset; if (yemax > 31) yemax = 31;
	int xemax = LeftMinMax[1].xe + offset; if (xemax > 31) xemax = 31;
	for (int ys = ysmin; ys <= ysmax; ys++)
	{
		for (int xs = xsmin; xs <= xsmax; xs++)
		{
			for (int ye = yemin; ye <= yemax; ye++)
			{
				for (int xe = xemin; xe <= xemax; xe++)
				{
					for (int ysdiv = 0; ysdiv < yblock; ysdiv++)
					{
						for (int xsdiv = 0; xsdiv < xblock; xsdiv++)
						{
							for (int yediv = 0; yediv < yblock; yediv++)
							{
								for (int xediv = 0; xediv < xblock; xediv++)
								{
									Table_tmp.id = id;
									Table_tmp.xs = xs + (xe - xs) * xsdiv / xblock;
									Table_tmp.ys = ys + (ye - ys) * ysdiv / yblock;
									Table_tmp.xe = xe - (xe - xs) * xediv / xblock;
									Table_tmp.ye = ye - (ye - ys) * yediv / yblock;
									FeatureLoc.push_back(Table_tmp);
									id++;
								}
							}
						} 
					}
				}
			}
		}
	}

	ysmin = RightMinMax[0].ys - offset; if (ysmin < 0) ysmin = 0;
	xsmin = RightMinMax[0].xs - offset; if (xsmin < 0) xsmin = 0;
	yemin = RightMinMax[0].ye - offset; if (yemin < 0) yemin = 0;
	xemin = RightMinMax[0].xe - offset; if (xemin < 0) xemin = 0;
	ysmax = RightMinMax[1].ys + offset; if (ysmax > 31) ysmax = 31;
	xsmax = RightMinMax[1].xs + offset; if (xsmax > 31) xsmax = 31;
	yemax = RightMinMax[1].ye + offset; if (yemax > 31) yemax = 31;
	xemax = RightMinMax[1].xe + offset; if (xemax > 31) xemax = 31;

	for (int ys = ysmin; ys <= ysmax; ys++)
	{
		for (int xs = xsmin; xs <= xsmax; xs++)
		{
			for (int ye = yemin; ye <= yemax; ye++)
			{
				for (int xe = xemin; xe <= xemax; xe++)
				{
					for (int ysdiv = 0; ysdiv < yblock; ysdiv++)
					{
						for (int xsdiv = 0; xsdiv < xblock; xsdiv++)
						{
							for (int yediv = 0; yediv < yblock; yediv++)
							{
								for (int xediv = 0; xediv < xblock; xediv++)
								{
									Table_tmp.id = id;
									Table_tmp.xs = xs + (xe - xs) * xsdiv / xblock;
									Table_tmp.ys = ys + (ye - ys) * ysdiv / yblock;
									Table_tmp.xe = xe - (xe - xs) * xediv / xblock;
									Table_tmp.ye = ye - (ye - ys) * yediv / yblock;

									FeatureLoc.push_back(Table_tmp);
									id++;
								}
							}
						}
					}
				}
			}
		}
	}
#ifdef DEBUG
	string FeatureLocFilename = WorkFolder + "FeatureLocation.txt";
	ofstream FeatureLocation(FeatureLocFilename.c_str());
	for (int i = 0; i < FeatureLoc.size();i++)
	{
		FeatureLocation << FeatureLoc[i].id << "	" << FeatureLoc[i].xs << "	" 
			<< FeatureLoc[i].ys << "	" << FeatureLoc[i].xe << "	" << FeatureLoc[i].ye << endl;
	}
#endif
}

int Type0(TableList box, int** integral)
{
	TableList negbox, posbox;
	negbox.ys = box.ys; negbox.xs = box.xs;                         negbox.ye = box.ye; negbox.xe = box.xs + (box.xe - box.xs) / 2;
	posbox.ys = box.ys; posbox.xs = box.xs + (box.xe - box.xs) / 2; posbox.ye = box.ye; posbox.xe = box.xe;
	int negRigion = integral[negbox.ys][negbox.xs] + integral[negbox.ye][negbox.xe] - integral[negbox.ys][negbox.xe] - integral[negbox.ye][negbox.xs];
	int posRigion = integral[posbox.ys][posbox.xs] + integral[posbox.ye][posbox.xe] - integral[posbox.ys][posbox.xe] - integral[posbox.ye][posbox.xs];
	return posRigion - negRigion;
}

int Type1(TableList box, int** integral)
{
	TableList negbox, posbox;
	negbox.ys = box.ys;                         negbox.xs = box.xs; negbox.ye = box.ys + (box.ye - box.ys) /2; negbox.xe = box.xe;
	posbox.ys = box.ys + (box.ye - box.ys) / 2; posbox.xs = box.xs; posbox.ye = box.ye;                        posbox.xe = box.xe;
	int negRigion = integral[negbox.ys][negbox.xs] + integral[negbox.ye][negbox.xe] - integral[negbox.ys][negbox.xe] - integral[negbox.ye][negbox.xs];
	int posRigion = integral[posbox.ys][posbox.xs] + integral[posbox.ye][posbox.xe] - integral[posbox.ys][posbox.xe] - integral[posbox.ye][posbox.xs];
	return posRigion - negRigion;
}

int Type2(TableList box, int** integral)
{
	TableList negbox1, negbox2, posbox1, posbox2;
	negbox1.ys = box.ys;                         negbox1.xs = box.xs;                         negbox1.ye = box.ys + (box.ye - box.ys) / 2; negbox1.xe = box.xs + (box.xe - box.xs) / 2;
	negbox2.ys = box.ys + (box.ye - box.ys) / 2; negbox2.xs = box.xs + (box.xe - box.xs) / 2; negbox2.ye = box.ye;                         negbox2.xe = box.xe;
	posbox1.ys = box.ys;                         posbox1.xs = box.xs + (box.xe - box.xs) / 2; posbox1.ye = box.ys + (box.ye - box.ys) / 2; posbox1.xe = box.xe;
	posbox2.ys = box.ys + (box.ye - box.ys) / 2; posbox2.xs = box.xs;                         posbox2.ye = box.ye;                         posbox2.xe = box.xs + (box.xe - box.xs) / 2;
	int negRigion1 = integral[negbox1.ys][negbox1.xs] + integral[negbox1.ye][negbox1.xe] - integral[negbox1.ys][negbox1.xe] - integral[negbox1.ye][negbox1.xs];
	int negRigion2 = integral[negbox2.ys][negbox2.xs] + integral[negbox2.ye][negbox2.xe] - integral[negbox2.ys][negbox2.xe] - integral[negbox2.ye][negbox2.xs];
	int posRigion1 = integral[posbox1.ys][posbox1.xs] + integral[posbox1.ye][posbox1.xe] - integral[posbox1.ys][posbox1.xe] - integral[posbox1.ye][posbox1.xs];
	int posRigion2 = integral[posbox2.ys][posbox2.xs] + integral[posbox2.ye][posbox2.xe] - integral[posbox2.ys][posbox2.xe] - integral[posbox2.ye][posbox2.xs];
	return posRigion1 + posRigion2 - negRigion1 - negRigion2;
}

int Type3(TableList box, int** integral)
{
	TableList negbox1, posbox1, posbox2, posbox3, posbox4;
	negbox1.ys = box.ys + (box.ye - box.ys) / 4;     negbox1.xs = box.xs + (box.xe - box.xs) / 4;     negbox1.ye = box.ys + (box.ye - box.ys) * 3 / 4; negbox1.xe = box.xs + (box.xe - box.xs) * 3 / 4;
	posbox1.ys = box.ys;                             posbox1.xs = box.xs;                             posbox1.ye = box.ye;                             posbox1.xe = box.xs + (box.xe - box.xs) / 4;
	posbox2.ys = box.ys;                             posbox2.xs = box.xs + (box.xe - box.xs) * 3 / 4; posbox2.ye = box.ye;                             posbox2.xe = box.xe;
	posbox3.ys = box.ys;                             posbox3.xs = box.xs + (box.xe - box.xs) / 4;     posbox3.ye = box.ys + (box.ye - box.ys) / 4;     posbox3.xe = box.xs + (box.xe - box.xs) * 3 / 4;
	posbox4.ys = box.ys + (box.ye - box.ys) * 3 / 4; posbox4.xs = box.xs + (box.xe - box.xs) / 4;     posbox4.ye = box.ye;                             posbox4.xe = box.xs + (box.xe - box.xs) * 3 / 4;
	int negRigion1 = integral[negbox1.ys][negbox1.xs] + integral[negbox1.ye][negbox1.xe] - integral[negbox1.ys][negbox1.xe] - integral[negbox1.ye][negbox1.xs];
	int posRigion1 = integral[posbox1.ys][posbox1.xs] + integral[posbox1.ye][posbox1.xe] - integral[posbox1.ys][posbox1.xe] - integral[posbox1.ye][posbox1.xs];
	int posRigion2 = integral[posbox2.ys][posbox2.xs] + integral[posbox2.ye][posbox2.xe] - integral[posbox2.ys][posbox2.xe] - integral[posbox2.ye][posbox2.xs];
	int posRigion3 = integral[posbox3.ys][posbox3.xs] + integral[posbox3.ye][posbox3.xe] - integral[posbox3.ys][posbox3.xe] - integral[posbox3.ye][posbox3.xs];
	int posRigion4 = integral[posbox4.ys][posbox4.xs] + integral[posbox4.ye][posbox4.xe] - integral[posbox4.ys][posbox4.xe] - integral[posbox4.ye][posbox4.xs];
	return posRigion1 + posRigion2 + posRigion3 + posRigion4 - negRigion1;
}

/*For each image get its features*/
void AllImageFeature(vector <TableList> FeatureLoc, int** img, int** integral, 
	                 vector <FeatureValue> &ImageFeature, int k, int img_cnt, 
	                 vector <TableList> LeftTable, vector <TableList> RightTable)
{
	FeatureValue vtmp;
	int j;
	for (int i = 0; i < FeatureLoc.size() * 4; i++)
	{
		vtmp.box = FeatureLoc[i / 4];
		vtmp.box.id = k;
		if (i % 4 == 0) vtmp.fv = Type0(FeatureLoc[i / 4], integral);
		else if (i % 4 == 1) vtmp.fv = Type1(FeatureLoc[i / 4], integral);
		else if (i % 4 == 2) vtmp.fv = Type2(FeatureLoc[i / 4], integral);
		else vtmp.fv = Type3(FeatureLoc[i / 4], integral);

		vtmp.hit = 0;
		j = 0;
		while (j < img_cnt && j < LeftTable.size() && LeftTable[j].id != k) j++;
		if (j < LeftTable.size() && LeftTable[j].id == k &&
			vtmp.box.xs >= LeftTable[j].xs &&
			vtmp.box.ys >= LeftTable[j].ys &&
			vtmp.box.xe <= LeftTable[j].xe &&
			vtmp.box.ye <= LeftTable[j].ye) vtmp.hit = 1;

		j = 0;
		while (j < img_cnt && j < RightTable.size() && RightTable[j].id != k) j++;
		if (j < RightTable.size() && RightTable[j].id == k && vtmp.hit == 0 &&
			vtmp.box.xs >= RightTable[j].xs &&
			vtmp.box.ys >= RightTable[j].ys &&
			vtmp.box.xe <= RightTable[j].xe &&
			vtmp.box.ye <= RightTable[j].ye) vtmp.hit = 1;

 		ImageFeature.push_back(vtmp);
	}
}

void BuildFeatureImage(string FeatureListFilename, string FeatureImageFilename, vector <TableList> FeatureLoc,
	int img_start, int img_end)
{
	ifstream FeatureList(FeatureListFilename.c_str(), std::ifstream::binary);
	ofstream FeatureImageList(FeatureImageFilename.c_str(), std::ofstream::binary);
	FeatureValue Feature_tmp;
	int FeatureLen = FeatureLoc.size() * 4;

	for (int fid = 0; fid < FeatureLen; fid++)
	{
#ifdef DEBUG
		stringstream ss;
		ss << fid;
		string FeatureidFilename = WorkFolder + "Feature" + ss.str() + ".txt";
		ofstream FeatureImage(FeatureidFilename.c_str());
#endif
		FeatureList.clear();
		FeatureList.seekg(fid * sizeof(FeatureValue), ios::beg);

		for (int img = img_start; img < img_end; img++)
		{
			FeatureList.read((char*)& Feature_tmp, sizeof(FeatureValue));
			FeatureImageList.write((char*)& Feature_tmp, sizeof(FeatureValue));
			FeatureList.seekg((FeatureLen - 1) * sizeof(FeatureValue), ios::cur);

#ifdef DEBUG
			FeatureImage << Feature_tmp.box.id << "	" << Feature_tmp.box.xs << "	" << Feature_tmp.box.ys << "	"
				<< Feature_tmp.box.xe << "	" << Feature_tmp.box.ye << "	" << Feature_tmp.fv << "	" << Feature_tmp.hit << endl;
#endif
		}
#ifdef DEBUG
		FeatureImage.close();
#endif
	}
}

void MergeFeatureImage(string OldFeatureImageFilename, string NewFeatureImageFilename, vector <TableList> FeatureLoc,
	int img_start, int img_end)
{
	ifstream FeatureImage0(OldFeatureImageFilename.c_str(), std::ifstream::binary);
	ifstream FeatureImage1(NewFeatureImageFilename.c_str(), std::ifstream::binary);
	string AllFeatureImageFilename = OldFeatureImageFilename + "All";
	ofstream outFeatureImage(AllFeatureImageFilename.c_str(), std::ofstream::binary);
	FeatureValue Feature_tmp;
	vector <FeatureValue> Feature_tmp0(img_start), Feature_tmp1(img_end - img_start);

	int FeatureLen = FeatureLoc.size() * 4;

	for (int fid = 0; fid < FeatureLen; fid++)
	{
#ifdef DEBUG
		stringstream ss;
		ss << fid;
		string FeatureidFilename = WorkFolder + "MFeature" + ss.str() + ".txt";
		ofstream FeatureImage(FeatureidFilename.c_str());
#endif

		FeatureImage0.read((char*)& Feature_tmp0[0], sizeof(FeatureValue) * img_start);
		outFeatureImage.write((char*)& Feature_tmp0[0], sizeof(FeatureValue) * img_start);

#ifdef DEBUG
		for (int img = 0; img < img_start; img++)
		{
			FeatureImage << Feature_tmp0[img].box.id << "	" << Feature_tmp0[img].box.xs << "	" << Feature_tmp0[img].box.ys << "	"
				<< Feature_tmp0[img].box.xe << "	" << Feature_tmp0[img].box.ye << "	" << Feature_tmp0[img].fv << "	" << Feature_tmp0[img].hit << endl;
		}
#endif
		FeatureImage1.read((char*)& Feature_tmp1[0], sizeof(FeatureValue) * (img_end - img_start));
		outFeatureImage.write((char*)& Feature_tmp1[0], sizeof(FeatureValue) * (img_end - img_start));

#ifdef DEBUG
		for (int img = img_start; img < img_end; img++)
		{
				FeatureImage << Feature_tmp1[img - img_start].box.id << "	" << Feature_tmp1[img - img_start].box.xs << "	" << Feature_tmp1[img - img_start].box.ys << "	"
				<< Feature_tmp1[img - img_start].box.xe << "	" << Feature_tmp1[img - img_start].box.ye << "	" << Feature_tmp1[img - img_start].fv << "	" << Feature_tmp1[img - img_start].hit << endl;
		}
		FeatureImage.close();
#endif
		/*
		for (int img = 0; img < img_start; img++)
		{
			FeatureImage0.read((char*)& Feature_tmp, sizeof(FeatureValue));
			outFeatureImage.write((char*)& Feature_tmp, sizeof(FeatureValue));
		}
		for (int img = img_start; img < img_end; img++)
		{
			FeatureImage1.read((char*)& Feature_tmp, sizeof(FeatureValue));
			outFeatureImage.write((char*)& Feature_tmp, sizeof(FeatureValue));
		}
		*/
	}
	FeatureImage0.close(); 
	FeatureImage1.close(); 
	outFeatureImage.close();
}
