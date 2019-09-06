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

/*Read and verify BMP input image and return the header as a char pointer array*/
char* ReadBMP256Size(std::string, int*, int*, int*);

/*Read in the BMP image and return the average of the pixel values*/
int ReadBMP256(std::string, int, int, int, int**);

/*Write to BMP img file*/
void WriteBMP256(std::string, int, int, int, int**, char*);
