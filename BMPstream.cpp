#include "BMPstream.h"
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
#include <iterator>
#include "features.h"

using namespace std;
#define DEBUG_FEATURE

vector<char> ReadBMP256Size(string filename, int* width, int* height, int* offset)
{
	ifstream bmp256(filename.c_str(), std::ios::in | ::ifstream::binary);

	if (bmp256.is_open())
	{
		char buffer[54];
		bmp256.read(buffer, 54);
		if ((buffer[0] != 'B') || (buffer[1] != 'M'))
		{
			cout << "This is not a BMP file!" << endl;
			bmp256.close();
			return;
		}

		*width = *(int*)& buffer[18];
		*height = *(int*)& buffer[22];
		int bits = *(int*)& buffer[28];
		int color = *(int*)& buffer[46];

		if (bits <= 8) *offset = color * 4;
		*offset += 54;

	}
	vector<char> header;
	header.reserve(*offset);	//Constant time with reserve
	copy(istream_iterator<char>(bmp256), istream_iterator<char>(), back_inserter(header));
	// // char* tmp = new char[*offset];
	// // bmp256.seekg(0, bmp256.beg);
	// // bmp256.read(tmp, *offset);
	// for(int i = 0; i < *offset; i++) header.push_back(tmp[i]);
	// delete[] tmp;
	bmp256.close();
	return header;
}

int ReadBMP256(string filename, int width, int height, int offset, int** img)
{
	ifstream bmp256(filename.c_str(), std::ifstream::binary);
	int sum = 0;

	if (bmp256)
	{
		bmp256.seekg(0, bmp256.end);
		int length = bmp256.tellg();
		bmp256.seekg(0, bmp256.beg);
		char* buffer = new char[length];
		bmp256.read(buffer, length);

		if ((buffer[0] != 'B') || (buffer[1] != 'M'))
		{
			cout << "This is not a BMP file!" << endl;
			bmp256.close();
			return 0;
		}

		int i = offset;
		for (int row = height - 1; row >= 0; row--) {
			for (int col = 0; col < width; col++) {
				unsigned char value = (unsigned char) buffer[i];
				int pixel = (int) value;
				img[row][col] = pixel;
				sum += pixel;
				i++;
			}
		}
	}
	bmp256.close();
	return sum / width / height;
}

void WriteBMP256(string filename, int width, int height, int offset, int** img, char* header)
{
	ofstream bmp256(filename.c_str(), std::ofstream::binary);
	if (bmp256)
	{
		bmp256.write(header, offset);
		char* buffer = new char[width * height];

		int i = 0;
		for (int row = height - 1; row >= 0; row--) {
			for (int col = 0; col < width; col++) {
				buffer[i] = (char) img[row][col];
				i++;
			}
		}

		bmp256.write(buffer, width * height);
	}
	bmp256.close();
}