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

/*Read in the correct eye-table data and use it as the min-max constraints for feature range for each image, separated as left and right eyes*/
void GetEyeList(std::string, std::vector <TableList>&, std::vector <TableList>&, 
                std::vector <TableList>&, std::vector <TableList>&, int);
