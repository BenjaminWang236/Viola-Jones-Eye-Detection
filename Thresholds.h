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

/*Find the thresholds (positive and negative) for each feature*/
void BuildFeatureThreshold(std::string, std::vector <TableList>, std::vector <FeatureThreshold>&, int);

/*For each feature find its hit (0 or 1) if greater or equal to positive threshold and less than or equal to negative threshold across all (imgcnt) images*/
void BuildThresholdHit(std::string, std::string, std::vector <FeatureThreshold>, int);
