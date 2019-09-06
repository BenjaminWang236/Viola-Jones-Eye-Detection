#include <iostream>
#include <fstream>
#include <string>
#include <cstdlib>
#include <vector>
#include <sstream>
#include <cmath>
#include <algorithm>
// #include <iomanip>      // std::setw
#include <chrono>

#include "features.h"
#include "GetEyeList.h"
#include "BMPstream.h"
#include "imgProcessing.h"
#include "Thresholds.h"
#include "Weights.h"
#include "Train.h"

//#include <Magick++.h>

using namespace std;
//using namespace Magick;
#define DEBUG_FEATURE

string WorkFolder = "E:/Viola-Jones-Eye-Detection/";
string SourceEyeTableFilename = "eye_point_data.txt";

string TrainFolder = "trainimg3/";
string OutputFolder = "trained/";
string ImgPrefix = "trainimg_";

int imgsizeW = 32, imgsizeH = 32, img_cnt = 100;

int main(int argc, char** argv)
{
	auto start = std::chrono::high_resolution_clock::now();
	cout << "Started counting" << endl;

#ifdef DEBUG
	string ImageFilename = WorkFolder + "Image.txt";
	ofstream Image(ImageFilename.c_str());
	string NormalFilename = WorkFolder + "Normal.txt";
	ofstream Normal(NormalFilename.c_str());
	string IntegralFilename = WorkFolder + "Integral.txt";
	ofstream Integral(IntegralFilename.c_str());
#endif
	int img_cnt = 10;

	int img_start = 20, img_end = 29;
	bool BuildImageFeatureOnly = true;
	bool FromBuildFeatureThreshold = false;

	if (argc != 6)
	{
		// 0 1 0 79 80
		cout << "BuildFeatureOnly TrainOnly Start End Total";
		return 0;
	}
	else
	{
		if(atoi(argv[1]) == 1) BuildImageFeatureOnly = true;
		else if (atoi(argv[1]) == 0) BuildImageFeatureOnly = false;
		else 
		{
			cout << "BuildFeatureOnly TrainOnly Start End Total";
			return 0;
		}

		if (atoi(argv[2]) == 1) FromBuildFeatureThreshold = true;
		else if (atoi(argv[2]) == 0) FromBuildFeatureThreshold = false;
		else
		{
			cout << "BuildFeatureOnly TrainOnly Start End Total";
			return 0;
		}

		img_start = atoi(argv[3]); img_end = atoi(argv[4]); img_cnt = atoi(argv[5]);
		
		for (int i = 0; i < argc; i++) cout << " " << argv[i] << " ";
		cout << endl;
	}

	int imgsizeW, imgsizeH, offset;
	string bmpsource = WorkFolder + TrainFolder + ImgPrefix + "1.bmp";

	char* header = ReadBMP256Size(bmpsource, &imgsizeW, &imgsizeH, &offset);
	char* BMP256Header = new char[offset];
	for (int i = 0; i < offset; i++) BMP256Header[i] = header[i];

	int** img = (int**)malloc(sizeof(int) * imgsizeH);
	for (int i = 0; i < imgsizeH; i++) img[i] = (int*)malloc(sizeof(int) * imgsizeW);

	int** integral = (int**)malloc(sizeof(int) * imgsizeH);
	for (int i = 0; i < imgsizeH; i++) integral[i] = (int*)malloc(sizeof(int) * imgsizeW);

	string SourceEyeTable = WorkFolder + SourceEyeTableFilename;
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
		FeatureListFilename = WorkFolder + "ImageFeature" + ss.str() + ".bin";
		FeatureImageFilename = WorkFolder + "FeatureImage" + ss.str() + ".bin";
		OldFeatureImageFilename = WorkFolder + "FeatureImage.bin";
		img_end++;
		cout << "After this program finished, Please ...." << endl;
		cout << "del " << OldFeatureImageFilename << endl;
		cout << "move " << OldFeatureImageFilename << "All " << OldFeatureImageFilename << endl;
		cout << "Or......" << endl;
		cout << "rm " << OldFeatureImageFilename << endl;
		cout << "mv " << OldFeatureImageFilename << "All " << OldFeatureImageFilename << endl;
	} 
	else
	{
		img_start = 0; 
		img_end = img_cnt; 
		FeatureListFilename = WorkFolder + "ImageFeature.bin";
		FeatureImageFilename = WorkFolder + "FeatureImage.bin";

		if (FromBuildFeatureThreshold)
		{
			cout << "Make sure do below copy file before training................." << endl;
			cout << "copy " << FeatureImageFilename << " " << WorkFolder + "AllFeatureImage.bin" << endl;
			cout << "Or......" << endl;
			cout << "cp " << FeatureImageFilename << " " << WorkFolder + "AllFeatureImage.bin" << endl;
		}
	}

	if (!FromBuildFeatureThreshold)
	{
		ofstream FeatureList(FeatureListFilename.c_str(), std::ofstream::binary);
		for (int k = img_start; k < img_end; k++)
		{
			stringstream ss;
			ss << k;
			string bmpsource = WorkFolder + TrainFolder + ImgPrefix + ss.str() + ".bmp";

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
			for (int i = 0; i < ImageFeature.size();i++) FeatureList.write((char*)& ImageFeature[i], sizeof(FeatureValue));

#ifdef DEBUG	
			string FeaturetableFilename = WorkFolder + "Image" + to_string(k) + "Feature.txt";
			ofstream FeatureTable(FeaturetableFilename.c_str());
			for (int i = 0; i < ImageFeature.size();i++)
				FeatureTable << ImageFeature[i].box.id << "	" << ImageFeature[i].box.xs << "	"
				<< ImageFeature[i].box.ys << "	" << ImageFeature[i].box.xe << "	"
				<< ImageFeature[i].box.ye << "	" << ImageFeature[i].fv << "	" << ImageFeature[i].hit << endl;
			FeatureTable.close();
#endif
		}
		FeatureList.close();
		for (int i = 0; i < imgsizeH; i++) free(img[i]); free(img);
		for (int i = 0; i < imgsizeH; i++) free(integral[i]); free(integral);
		LeftTable.clear(); RightTable.clear(); ImageFeature.clear();
	}
#ifdef DEBUG
	Image.close();
	Normal.close();
	Integral.close();
#endif
	if (!FromBuildFeatureThreshold) BuildFeatureImage(FeatureListFilename, FeatureImageFilename, FeatureLoc, img_start, img_end);
	if (BuildImageFeatureOnly) MergeFeatureImage(OldFeatureImageFilename, FeatureImageFilename, FeatureLoc, img_start, img_end);

	if (!BuildImageFeatureOnly || FromBuildFeatureThreshold)
	{
		if(FromBuildFeatureThreshold) FeatureImageFilename = WorkFolder + "AllFeatureImage.bin";
		vector <FeatureThreshold> ThresholdTable;
		BuildFeatureThreshold(FeatureImageFilename, FeatureLoc, ThresholdTable, img_cnt);
		FeatureLoc.clear();

#ifdef DEBUG
		string ThresholdTableFilename = WorkFolder + "ThresholdList.txt";
		ofstream ThresholdList(ThresholdTableFilename.c_str());
		for (int i = 0; i < ThresholdTable.size();i++)
		{
			ThresholdList << ThresholdTable[i].box.id << "	" << ThresholdTable[i].box.xs << "	"
				<< ThresholdTable[i].box.ys << "	" << ThresholdTable[i].box.xe << "	"
				<< ThresholdTable[i].box.ye << "	" << ThresholdTable[i].thp << "	" << ThresholdTable[i].thn << endl;
		}
		ThresholdList.close();
#endif

		double** weights = (double**)malloc(sizeof(double) * ThresholdTable.size());
		for (int i = 0; i < ThresholdTable.size(); i++) weights[i] = (double*)malloc(sizeof(double) * img_cnt);

		vector <int> ThresholdHit;
		BuildThresholdHit(FeatureImageFilename, ThresholdHit, ThresholdTable, img_cnt);

#ifdef DEBUG
		string ThresholdHitFilename = WorkFolder + "ThresholdHitList.txt";
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

		initWeights(FeatureImageFilename, weights, ThresholdTable, img_cnt);

		string OutputTable = WorkFolder + "TrainTable.txt";
		ofstream TableOut(OutputTable.c_str());

		vector <int> MinIndex;

		TableOut << "INDEX TYPE XSTART YSTART XEND YEND THRESHOLD_P THRESHOLD_N Alpha" << endl;

		for (int i = 0; i < ThresholdTable.size(); i++)
		{
			normWeights(weights, ThresholdTable, img_cnt);
			TrainOut minidx_beta = train(TableOut, weights, ThresholdHit, FeatureImageFilename, ThresholdTable, MinIndex, img_cnt);
			MinIndex.push_back(minidx_beta.minidx);
			updateWeights(weights, ThresholdHit, FeatureImageFilename, ThresholdTable, img_cnt, minidx_beta.minidx, minidx_beta.beta);
		}

		for (int i = 0; i < ThresholdTable.size(); i++) free(weights[i]); free(weights);
		ThresholdTable.clear(); ThresholdHit.clear();
	}

	cout << "Finished counting" << endl;
	auto end = std::chrono::high_resolution_clock::now();
	auto duration0 = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
	auto duration = end-start;
	// auto microsec = std::chrono::duration_cast<std::chrono::microseconds>(duration);
	cout << "--- " << "Execution time: " << duration0.count() << " microseconds" << " ---" << endl;
	cout << "--- " << "Execution time: " << (std::chrono::duration_cast<std::chrono::hours>(duration)).count() << "::" << (std::chrono::duration_cast<std::chrono::minutes>(duration)).count() << "::"
	<< (std::chrono::duration_cast<std::chrono::seconds>(duration)).count() << ":: " << (std::chrono::duration_cast<std::chrono::milliseconds>(duration)).count() << endl;
	
	return 0;

}


// /*main*/
// int main()
// {
// 	auto start = std::chrono::high_resolution_clock::now();
// 	cout << "Started counting" << endl;
// #ifdef DEBUG
// 	string ImageFilename = WorkFolder + "Image.txt";
// 	ofstream Image(ImageFilename.c_str());
// 	string NormalFilename = WorkFolder + "Normal.txt";
// 	ofstream Normal(NormalFilename.c_str());
// 	string IntegralFilename = WorkFolder + "Integral.txt";
// 	ofstream Integral(IntegralFilename.c_str());
// #endif

// 	int imgsizeW, imgsizeH, offset, file_id = 0;
// 	string bmpsource = WorkFolder + TrainFolder + ImgPrefix + "1.bmp";

// 	char* header = ReadBMP256Size(bmpsource, &imgsizeW, &imgsizeH, &offset);
// 	char* BMP256Header = new char[offset];
// 	for (int i = 0; i < offset; i++) BMP256Header[i] = header[i];

// 	int** img = (int**)malloc(sizeof(int) * imgsizeH);
// 	for (int i = 0; i < imgsizeH; i++) img[i] = (int*)malloc(sizeof(int) * imgsizeW);

// 	int** integral = (int**)malloc(sizeof(int) * imgsizeH);
// 	for (int i = 0; i < imgsizeH; i++) integral[i] = (int*)malloc(sizeof(int) * imgsizeW);

// 	string SourceEyeTable = WorkFolder + SourceEyeTableFilename;
// 	vector <TableList> LeftTable, RightTable, LeftMinMax, RightMinMax;
// 	GetEyeList(SourceEyeTable, imgsizeW, LeftTable, RightTable, LeftMinMax, RightMinMax);

// 	vector <TableList> FeatureLoc;
// 	BuildFeatureLoc(FeatureLoc, LeftMinMax, RightMinMax, imgsizeW);
// 	LeftMinMax.clear(); RightMinMax.clear();

// 	string FeatureListFilename = WorkFolder + "ImageFeature.bin";
// // 	ofstream FeatureList(FeatureListFilename.c_str(), std::ofstream::binary);

// // 	vector <FeatureValue> ImageFeature;
// // 	/*For each image generate its features*/
// // 	for (int k = 0; k < img_cnt; k++)
// // 	{
// // 		stringstream ss;
// // 		ss << file_id;
// // 		string bmpsource = WorkFolder + TrainFolder + ImgPrefix + ss.str() + ".bmp";
// // 		file_id++;

// // 		int avg = ReadBMP256(bmpsource, imgsizeW, imgsizeH, offset, img);

// // #ifdef DEBUG
// // 		cout << "average of image " << k << " = " << avg << endl;
// // 		for (int row = 0; row < imgsizeH; row++)
// // 		{
// // 			for (int col = 0; col < imgsizeW; col++)
// // 			{
// // 				Image << img[row][col] << "	";
// // 			}
// // 			Image << endl;
// // 		}
// // #endif
// // 		normalize(imgsizeW, imgsizeH, img, avg, 8); //keep 8 bits of floating point

// // 		integralAll(img, integral, imgsizeW, imgsizeH);

// // #ifdef DEBUG
// // 		for (int row = 0; row < imgsizeH; row++)
// // 		{
// // 			for (int col = 0; col < imgsizeW; col++)
// // 			{
// // 				Normal << img[row][col] << "	";
// // 				Integral << integral[row][col] << "	";
// // 			}
// // 			Normal << endl;
// // 			Integral << endl;
// // 		}
// // #endif
// // 		ImageFeature.clear();
// // 		AllImageFeature(FeatureLoc, img, integral, ImageFeature, k, img_cnt, LeftTable, RightTable);
// // 		for (int i = 0; i < ImageFeature.size();i++) FeatureList.write((char*)& ImageFeature[i], sizeof(FeatureValue));

// // #ifdef DEBUG	
// // 		string FeaturetableFilename = WorkFolder + "Image" + to_string(k) + "Feature.txt";
// // 		ofstream FeatureTable(FeaturetableFilename.c_str());
// // 		for (int i = 0; i < ImageFeature.size();i++)
// // 			FeatureTable << ImageFeature[i].box.id << "	" << ImageFeature[i].box.xs << "	"
// // 			<< ImageFeature[i].box.ys << "	" << ImageFeature[i].box.xe << "	"
// // 			<< ImageFeature[i].box.ye << "	" << ImageFeature[i].fv << "	" << ImageFeature[i].hit << endl;
// // 		FeatureTable.close();
// // #endif
// // 	}
// // 	FeatureList.close();
// // 	for (int i = 0; i < imgsizeH; i++) free(img[i]); free(img);
// // 	for (int i = 0; i < imgsizeH; i++) free(integral[i]); free(integral);
// // 	LeftTable.clear(); RightTable.clear(); ImageFeature.clear();

// #ifdef DEBUG
// 	Image.close();
// 	Normal.close();
// 	Integral.close();
// #endif
// 	string FeatureImageFilename = WorkFolder + "FeatureImage.bin";
// 	vector <FeatureThreshold> ThresholdTable;
// 	BuildFeatureThreshold(FeatureListFilename, FeatureImageFilename, FeatureLoc, ThresholdTable, img_cnt);
// 	FeatureLoc.clear();

// #ifdef DEBUG
// 	string ThresholdTableFilename = WorkFolder + "ThresholdList.txt";
// 	ofstream ThresholdList(ThresholdTableFilename.c_str());
// 	for (int i = 0; i < ThresholdTable.size();i++)
// 	{
// 		ThresholdList << ThresholdTable[i].box.id << "	" << ThresholdTable[i].box.xs << "	"
// 			<< ThresholdTable[i].box.ys << "	" << ThresholdTable[i].box.xe << "	"
// 			<< ThresholdTable[i].box.ye << "	" << ThresholdTable[i].thp << "	" << ThresholdTable[i].thn << endl;
// 	}
// 	ThresholdList.close();
// #endif

// 	double** weights = (double**)malloc(sizeof(double) * ThresholdTable.size());
// 	for (int i = 0; i < ThresholdTable.size(); i++) weights[i] = (double*)malloc(sizeof(double) * img_cnt);

// 	vector <int> ThresholdHit;
// 	BuildThresholdHit(FeatureImageFilename, ThresholdHit, ThresholdTable, img_cnt);

// #ifdef DEBUG
// 	string ThresholdHitFilename = WorkFolder + "ThresholdHitList.txt";
// 	ofstream ThresholdHitList(ThresholdHitFilename.c_str());
// 	for (int fid = 0; fid < ThresholdTable.size(); fid++)
// 	{
// 		for (int imgid = 0; imgid < img_cnt; imgid++)
// 		{
// 			ThresholdHitList << ThresholdHit[fid * img_cnt + imgid] << "	";
// 		}
// 		ThresholdHitList << endl;
// 	}
// 	ThresholdHitList.close();
// #endif

// 	initWeights(FeatureImageFilename, weights, ThresholdTable, img_cnt);

// 	string OutputTable = WorkFolder + "TrainTable.txt";
// 	ofstream TableOut(OutputTable.c_str());

// 	vector <int> MinIndex;

// 	TableOut << "INDEX TYPE XSTART YSTART XEND YEND THRESHOLD_P THRESHOLD_N Alpha" << endl;

// 	for (int i = 0; i < ThresholdTable.size(); i++)
// 	{
// 		normWeights(weights, ThresholdTable, img_cnt);
// 		TrainOut minidx_beta = train(TableOut, weights, ThresholdHit, FeatureImageFilename, ThresholdTable, MinIndex, img_cnt);
// 		MinIndex.push_back(minidx_beta.minidx);
// 		updateWeights(weights, ThresholdHit, FeatureImageFilename, ThresholdTable, img_cnt, minidx_beta.minidx, minidx_beta.beta);
// 	}

// //	string bmpdest = WorkFolder + OutputFolder + ImgPrefix + to_string(file_id) + ".bmp";
// //	WriteBMP256(bmpdest, imgsizeW, imgsizeH, offset, img, BMP256Header);
// 	for (int i = 0; i < ThresholdTable.size(); i++) free(weights[i]); free(weights);
// 	ThresholdTable.clear(); ThresholdHit.clear();

// 	cout << "Finished counting" << endl;
// 	auto end = std::chrono::high_resolution_clock::now();
// 	auto duration0 = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
// 	auto duration = end-start;
// 	// auto microsec = std::chrono::duration_cast<std::chrono::microseconds>(duration);
// 	cout << "--- " << "Execution time: " << duration0.count() << " microseconds" << " ---" << endl;
// 	cout << "--- " << "Execution time: " << (std::chrono::duration_cast<std::chrono::hours>(duration)).count() << "::" << (std::chrono::duration_cast<std::chrono::minutes>(duration)).count() << "::"
// 	<< (std::chrono::duration_cast<std::chrono::seconds>(duration)).count() << ":: " << (std::chrono::duration_cast<std::chrono::milliseconds>(duration)).count() << endl;

// 	return 0;
// }

