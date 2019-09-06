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

/*Normalize the image by the average pixel value (Used to be max)*/
void normalize(int, int, int**, int, int);

/*Compute the integral image to save individual computations later*/
void integralAll(int**, int**, int, int);

/*Draw a white box on the image to mark the eye(s)*/
void makeBox(std::vector <int>, int**, int);
