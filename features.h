#pragma once
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

/*Baasic feature as a box with id and x/y start/ends*/
struct TableList {
	int id;
	int xs;
	int ys;
	int xe;
	int ye;
};

/*Feature with its basic box, feature value and hit(0 or 1)*/
struct FeatureValue {
	//TableList box;
	/*Feature value*/
	int fv;
	/*Number of hits*/
	int hit;
};

/*Feature with its basic box, positive/negative thresholds*/
struct FeatureThreshold {
	TableList box;
	/*Positive threshold*/
	int thp;
	/*Negative threshold*/
	int thn;
};

/*Training output holder*/
struct TrainOut {
	/*Minimum index*/
	int minidx;
	/*Beta value based on the minimum error at ^minidx*/
	double beta;

};

/*Generate the 4 types of features' locations based on constraints determined before*/
void BuildFeatureLoc(std::vector <TableList>&, std::vector <TableList>, std::vector <TableList>, int);

/*Compute Haar-like feature 0's value*/
int Type0(TableList, int**);

/*Compute Haar-like feature 1's value*/
int Type1(TableList, int**);

/*Compute Haar-like feature 2's value*/
int Type2(TableList, int**);

/*Compute Haar-like feature 3's value*/
int Type3(TableList, int**);

/*For each image get its features*/
void AllImageFeature(std::vector <TableList>, int**, int**, 
	                 std::vector <FeatureValue> &, int, int, 
	                 std::vector <TableList>, std::vector <TableList>);

/*Sort the features from all images in feature-order*/
void BuildFeatureImage(std::string, std::string, std::vector <TableList>, int, int);

/*Merge the corresponding features on two sets of images (probably continuous)*/
void MergeFeatureImage(std::string, std::string, std::vector <TableList>, int, int);
