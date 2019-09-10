#pragma once
#include <iostream>
#include <fstream>
#include <string>
#include <cstdlib>
#include <vector>
#include <sstream>
#include <cmath>
#include <algorithm>
#include <iomanip> // std::setw
#include <chrono>
#include "features.h"

/*Initialize the weight*/
void initWeights(std::string, std::string, int, int);

/*Normalize the weights so they add up to 1 as distributed weights*/
void normWeights(std::string, std::string, int, int);

/*Update the weights at the end of an iteration*/
void updateWeights(std::string, std::string, std::string, std::string, int, int, int, double);
