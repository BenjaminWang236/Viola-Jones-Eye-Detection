#include "imgProcessing.h"
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

using namespace std;
#define DEBUG_FEATURE

void normalize(int width, int height, int** image, int normal, int keep)
{
	int mul = 0x01 << keep;
	for (int row = 0; row < height; row++)
	{
		for (int col = 0; col < width; col++)
		{
			int p = image[row][col];
			int q = image[row][col] * mul / normal;
			image[row][col] = image[row][col] * mul / normal;
		}
	}
}

void integralAll(int** image, int** integral, int width, int height)
{
	// int** rowsum = (int**)malloc(sizeof(int) * height);
	// for (int i = 0; i < height; i++) rowsum[i] = (int*)malloc(sizeof(int) * width);
	int** rowsum = new int*[height];
	for(int i = 0; i < height; i++) rowsum[i] = new int[width];

	for (int row = 0; row < height; row++)
	{
		for (int col = 0; col < width; col++)
		{
			rowsum[row][col] = (col-1 >= 0) ? rowsum[row][col-1] + image[row][col] : image[row][col];
			integral[row][col] = (row-1 >= 0) ? integral[row-1][col] + rowsum[row][col] : rowsum[row][col];
		}
	}
}

void makeBox(vector <int> box, int** img, int imgcnt)
{
	int startX = box[imgcnt * 4];
	int startY = box[imgcnt * 4 + 1];
	int endX = box[imgcnt * 4 + 2];
	int endY = box[imgcnt * 4 + 3];
	for (int row = startX; row <= endX; row++)
	{
		img[startY][row] = 255;
		img[endY][row] = 255;
	}

	for (int col = startY; col <= endY; col++)
	{
		img[col][startX] = 255;
		img[col][endX] = 255;
	}
}

/*unsigned int getIntegral(int** image, int cornerWidth, int cornerHeight)
{
	int sum = 0;
	for (int row = 0; row <= cornerHeight; row++)
	{
		for (int col = 0; col <= cornerWidth; col++)
		{
			sum += image[row][col];
		}
	}
	return sum;
}

void integralAll(int** image, int** integral, int width, int height)
{

	for (int row = 0; row < height; row++)
	{
		for (int col = 0; col < width; col++)
		{
			integral[row][col] = getIntegral(image, col, row);
		}
	}
}*/
