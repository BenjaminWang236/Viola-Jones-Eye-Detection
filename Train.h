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

/*Takes std::chrono::duration and return it as readable hour::minutes::seconds::milliseconds::microseconds format*/
std::string format_duration(long);

/*Training the features on the BMP images*/
TrainOut train(std::ofstream&, std::string, std::string, std::string, std::vector <FeatureThreshold>, std::vector <int>, int);
