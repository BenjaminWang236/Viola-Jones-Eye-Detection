#include <iostream>
#include <fstream>
#include <string>
#include <cstdlib>
#include <vector>
#include <sstream>
#include <cmath>
#include <algorithm>
#include <iterator>
#include <chrono>

#include "features.h"
#include "GetEyeList.h"
#include "BMPstream.h"
#include "imgProcessing.h"
#include "Thresholds.h"
#include "Weights.h"
#include "Train.h"

using namespace std;
using namespace std::chrono;
#define DEBUG_FEATURE

// string WorkFolder = "E:/GitHub/Viola-Jones-Eye-Detection/";
string WorkFolder = "/Users/infin/OneDrive/NeuronBasic/Viola-Jones-Eye-Detection/";
string SourceEyeTableFilename = "eye_point_data.txt";

string TrainFolder = "trainimg3/";
string OutputFolder = "trained/";
string ImgPrefix = "trainimg_";

int imgsizeW = 32, imgsizeH = 32, img_cnt = 100;

int main(int argc, char** argv)
{
	auto start = high_resolution_clock::now();
#ifdef DEBUG
	string ImageFilename = WorkDrive + WorkFolder + "Image.txt";
	ofstream Image(ImageFilename.c_str());
	string NormalFilename = WorkDrive + WorkFolder + "Normal.txt";
	ofstream Normal(NormalFilename.c_str());
	string IntegralFilename = WorkDrive + WorkFolder + "Integral.txt";
	ofstream Integral(IntegralFilename.c_str());
#endif
	string WorkDrive = "C:";

	int img_start = 20, img_end = 29;
	int img_cnt = img_end - img_start + 1;

	bool BuildImageFeatureOnly = true;
	bool FromBuildFeatureThreshold = false;

	if (argc != 6)
	{
		cout << "WorkDrive BuildFeatureOnly TrainOnly Start End";
		return 0;
	}
	else
	{
		WorkDrive = argv[1];

		if(atoi(argv[2]) == 1) BuildImageFeatureOnly = true;
		else if (atoi(argv[2]) == 0) BuildImageFeatureOnly = false;
		else 
		{
			cout << "BuildFeatureOnly TrainOnly Start End Total";
			return 0;
		}

		if (atoi(argv[3]) == 1) FromBuildFeatureThreshold = true;
		else if (atoi(argv[3]) == 0) FromBuildFeatureThreshold = false;
		else
		{
			cout << "BuildFeatureOnly TrainOnly Start End Total";
			return 0;
		}

		img_start = atoi(argv[4]); img_end = atoi(argv[5]); img_cnt = img_end - img_start + 1;
		
		for (int i = 0; i < argc; i++) cout << " " << argv[i] << " ";
		cout << endl;
	}

	int imgsizeW, imgsizeH, offset;
	string bmpsource = WorkDrive + WorkFolder + TrainFolder + ImgPrefix + "1.bmp";

	vector<char> header;
	header = ReadBMP256Size(bmpsource, &imgsizeW, &imgsizeH, &offset);
	char* BMP256Header = new char[offset];
	for (int i = 0; i < offset; i++) BMP256Header[i] = header[i];

	int** img = (int**)malloc(sizeof(int) * imgsizeH);
	for (int i = 0; i < imgsizeH; i++) img[i] = (int*)malloc(sizeof(int) * imgsizeW);

	int** integral = (int**)malloc(sizeof(int) * imgsizeH);
	for (int i = 0; i < imgsizeH; i++) integral[i] = (int*)malloc(sizeof(int) * imgsizeW);

	string SourceEyeTable = WorkDrive + WorkFolder + SourceEyeTableFilename;
	vector <TableList> LeftTable, RightTable, LeftMinMax, RightMinMax;
	GetEyeList(SourceEyeTable, LeftTable, RightTable, LeftMinMax, RightMinMax, imgsizeW);

	vector <TableList> FeatureLoc;
	BuildFeatureLoc(FeatureLoc, LeftMinMax, RightMinMax, imgsizeW);
	LeftMinMax.clear(); RightMinMax.clear();

	string FeatureListFilename, FeatureImageFilename, OldFeatureImageFilename;
	vector <FeatureValue> ImageFeature;

	if (BuildImageFeatureOnly)
	{
		stringstream ss;
		ss << img_start << "_" << img_end;
		FeatureImageFilename = WorkDrive + WorkFolder + "FeatureImage" + ss.str() + ".bin";
		OldFeatureImageFilename = WorkDrive + WorkFolder + "FeatureImage.bin";
		cout << "After this program finished, Please ...." << endl;
		cout << "del " << OldFeatureImageFilename << endl;
		cout << "move " << OldFeatureImageFilename << "All " << OldFeatureImageFilename << endl;
		cout << "Or......" << endl;
		cout << "rm " << OldFeatureImageFilename << endl;
		cout << "mv " << OldFeatureImageFilename << "All " << OldFeatureImageFilename << endl;
	} 
	else
	{
		FeatureImageFilename = WorkDrive + WorkFolder + "FeatureImage.bin";

		if (FromBuildFeatureThreshold)
		{
			cout << "Make sure do below copy file before training................." << endl;
			cout << "copy " << FeatureImageFilename << " " << WorkDrive + WorkFolder + "AllFeatureImage.bin" << endl;
			cout << "Or......" << endl;
			cout << "cp " << FeatureImageFilename << " " << WorkDrive + WorkFolder + "AllFeatureImage.bin" << endl;
		}
	}

	if (!FromBuildFeatureThreshold)
	{
		cout << "Calculate image : ";
		for (int k = img_start; k <= img_end; k++)
		{
			stringstream ss;
			ss << k;
			string bmpsource = WorkDrive + WorkFolder + TrainFolder + ImgPrefix + ss.str() + ".bmp";

			int avg = ReadBMP256(bmpsource, imgsizeW, imgsizeH, offset, img);

#ifdef DEBUG
			cout << "average of image " << k << " = " << avg << endl;
			for (int row = 0; row < imgsizeH; row++)
			{
				for (int col = 0; col < imgsizeW; col++)
				{
					Image << img[row][col] << "	";
				}
				Image << endl;
			}
#endif
			normalize(imgsizeW, imgsizeH, img, avg, 8); //keep 8 bits of floating point

			integralAll(img, integral, imgsizeW, imgsizeH);

#ifdef DEBUG
			for (int row = 0; row < imgsizeH; row++)
			{
				for (int col = 0; col < imgsizeW; col++)
				{
					Normal << img[row][col] << "	";
					Integral << integral[row][col] << "	";
				}
				Normal << endl;
				Integral << endl;
			}
#endif
			ImageFeature.clear();
			AllImageFeature(FeatureLoc, img, integral, ImageFeature, k, img_cnt, LeftTable, RightTable);
			FeatureListFilename = WorkDrive + WorkFolder + "ImageFeature" + ss.str() + ".bin";
			ofstream FeatureList(FeatureListFilename.c_str(), std::ofstream::binary);
			FeatureList.write((char*)& ImageFeature[0], ImageFeature.size() * sizeof(FeatureValue));
			FeatureList.close();
			cout << k << " ";

#ifdef DEBUG
			string FeaturetableFilename = WorkDrive + WorkFolder + "Image" + to_string(k) + "Feature.txt";
			ofstream FeatureTable(FeaturetableFilename.c_str());
			for (int i = 0; i < ImageFeature.size();i++)
				FeatureTable << ImageFeature[i].box.id << "	" << ImageFeature[i].box.xs << "	"
				<< ImageFeature[i].box.ys << "	" << ImageFeature[i].box.xe << "	"
				<< ImageFeature[i].box.ye << "	" << ImageFeature[i].fv << "	" << ImageFeature[i].hit << endl;
			FeatureTable.close();
#endif
		}
		cout << endl;
		for (int i = 0; i < imgsizeH; i++) free(img[i]); free(img);
		for (int i = 0; i < imgsizeH; i++) free(integral[i]); free(integral);
		LeftTable.clear(); RightTable.clear(); ImageFeature.clear();
	}
#ifdef DEBUG
	Image.close();
	Normal.close();
	Integral.close();
#endif
	FeatureListFilename = WorkDrive + WorkFolder + "ImageFeature";
	if (!FromBuildFeatureThreshold)
	{
		BuildFeatureImage(FeatureListFilename, FeatureImageFilename, FeatureLoc, img_start, img_end);

		for (int i = img_start; i <= img_end; i++)
		{
			stringstream ss;
			ss << i;
			string FeatureListFilename0 = FeatureListFilename + ss.str() + ".bin";
			std::remove(FeatureListFilename0.c_str());
		}
	}
	if (BuildImageFeatureOnly) MergeFeatureImage(OldFeatureImageFilename, FeatureImageFilename, FeatureLoc, img_start, img_end);

	cout << "Finding Threshold......." << endl;

	if (!BuildImageFeatureOnly || FromBuildFeatureThreshold)
	{
		if(FromBuildFeatureThreshold) FeatureImageFilename = WorkDrive + WorkFolder + "AllFeatureImage.bin";
		vector <FeatureThreshold> ThresholdTable;
		BuildFeatureThreshold(FeatureImageFilename, FeatureLoc, ThresholdTable, img_cnt);
		FeatureLoc.clear();

#ifdef DEBUG
		string ThresholdTableFilename = WorkDrive + WorkFolder + "ThresholdList.txt";
		ofstream ThresholdList(ThresholdTableFilename.c_str());
		for (int i = 0; i < ThresholdTable.size();i++)
		{
			ThresholdList << ThresholdTable[i].box.id << "	" << ThresholdTable[i].box.xs << "	"
				<< ThresholdTable[i].box.ys << "	" << ThresholdTable[i].box.xe << "	"
				<< ThresholdTable[i].box.ye << "	" << ThresholdTable[i].thp << "	" << ThresholdTable[i].thn << endl;
		}
		ThresholdList.close();
#endif
		int FeatureLen = ThresholdTable.size();

		//double** weights = (double**)malloc(sizeof(double) * ThresholdTable.size());
		//for (int i = 0; i < ThresholdTable.size(); i++) weights[i] = (double*)malloc(sizeof(double) * img_cnt);

//		vector <int> ThresholdHit;
		string ThresholdHitFilename = WorkDrive + WorkFolder + "ThresholdHit.bin";
		BuildThresholdHit(FeatureImageFilename, ThresholdHitFilename, ThresholdTable, img_cnt);

#ifdef DEBUG
		string ThresholdHitFilename = WorkDrive + WorkFolder + "ThresholdHitList.txt";
		ofstream ThresholdHitList(ThresholdHitFilename.c_str());
		for (int fid = 0; fid < ThresholdTable.size(); fid++)
		{
			for (int imgid = 0; imgid < img_cnt; imgid++)
			{
				ThresholdHitList << ThresholdHit[fid * img_cnt + imgid] << "	";
			}
			ThresholdHitList << endl;
		}
		ThresholdHitList.close();
#endif
		string Weight0Filename = WorkDrive + WorkFolder + "Weight0.bin";
		string WeightNormalFilename = WorkDrive + WorkFolder + "Weight.bin";
		initWeights(FeatureImageFilename, Weight0Filename, FeatureLen, img_cnt);

		string OutputTable = WorkDrive + WorkFolder + "TrainTable.txt";
		ofstream TableOut(OutputTable.c_str());

		vector <int> MinIndex;

		TableOut << "INDEX TYPE XSTART YSTART XEND YEND THRESHOLD_P THRESHOLD_N Alpha" << endl;

		cout << " Training on : ";
		for (int i = 0; i < FeatureLen; i++)
		{
			if(i==522)
			{
				int qq = 1;
			}
			normWeights(Weight0Filename, WeightNormalFilename, FeatureLen, img_cnt);
			TrainOut minidx_beta = train(TableOut, WeightNormalFilename, ThresholdHitFilename, FeatureImageFilename, ThresholdTable, MinIndex, img_cnt);
			MinIndex.push_back(minidx_beta.minidx);
			updateWeights(Weight0Filename, WeightNormalFilename, ThresholdHitFilename, FeatureImageFilename, FeatureLen, img_cnt, minidx_beta.minidx, minidx_beta.beta);
			//if (i % 1000 == 0 || i == FeatureLen - 1) cout << i << " ";
			cout << i << " ";
		}
		cout << endl;
		ThresholdTable.clear();
	}

	auto end = high_resolution_clock::now();
	auto duration = duration_cast<microseconds>(end-start);
	cout << "--- Execution time: "<< format_duration((long) duration.count()) << " (" << duration.count() << " microseconds) ---" << endl;
	
	return 0;

}
