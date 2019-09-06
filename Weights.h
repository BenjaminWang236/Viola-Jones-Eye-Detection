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

/*Initialize the weight*/
void initWeights(std::string, double**, std::vector <FeatureThreshold>, int);

/*Normalize the weights so they add up to 1 as distributed weights*/
void normWeights(double**, std::vector <FeatureThreshold>, int);

/*Update the weights at the end of an iteration*/
void updateWeights(double**, std::vector <int>, std::string, std::vector <FeatureThreshold>, int, int, double);
