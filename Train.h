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

/*Training the features on the BMP images*/
TrainOut train(std::ofstream&, double**, std::vector <int>,
	std::string, std::vector <FeatureThreshold>, std::vector <int>, int);
