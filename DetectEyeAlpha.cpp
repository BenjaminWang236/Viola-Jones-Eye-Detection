#include <iostream>
#include <fstream>
#include <string>
#include <cstdlib>
#include <vector>
#include <sstream>
#include <cmath>
#include <algorithm>    
#include <chrono>
// #include "timer.h"
//#include <array> 
//#include <iomanip>      // std::setw

//#include <Magick++.h>

using namespace std;
using namespace std::chrono;
//using namespace Magick;
#define DEBUG_FEATURE

/*
img_0.bmp etc for testing, trainimg_0.bmp etc for training.
/Desktop/CPP/Viola_Jones/ for Linux
/Users/infin/OneDrive/NeuronBasic/Viola-Jones-Eye-Detection/ for Windows
*/

string WorkFolder = "/Users/infin/OneDrive/NeuronBasic/Viola-Jones-Eye-Detection/";
string ImageFolder = "trainimg_0/";
string ImgOutFolder = "detected/";

string format_duration(long dur) {
	microseconds milsec(dur);
	auto ms = duration_cast<milliseconds>(milsec);		milsec -= duration_cast<microseconds>(ms);
	auto secs = duration_cast<seconds>(ms);				ms -= duration_cast<milliseconds>(secs);
	auto mins = duration_cast<minutes>(secs);			secs -= duration_cast<seconds>(mins);
	auto hour = duration_cast<hours>(mins);				mins -= duration_cast<minutes>(hour);

	stringstream ss;
	ss << hour.count() << "h " << mins.count() << "m " << secs.count() << "s " <<
			ms.count() << "ms " << milsec.count() << "\xE6s";
	return ss.str();
}


struct TableList {
	int id;
	int xs;
	int ys;
	int xe;
	int ye;
};

struct FeatureList {
	TableList box;
	int thp;
	int thn;
	float alpha;
};

void ReadBMP256Size(string filename, int* width, int* height, int* offset, vector <char> &header)
{
	ifstream bmp256(filename.c_str(), std::ifstream::binary);

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
	bmp256.clear();
	bmp256.seekg(0, bmp256.beg);
	char* buffer = new char[*offset];

	bmp256.read(buffer, *offset * sizeof(char));
	for (int i = 0; i < *offset; i++) header.push_back(buffer[i]);
	bmp256.close();
	delete[] buffer;

}


int ReadBMP256(string filename, int width, int height, int offset, int** img)
{
	ifstream bmp256(filename.c_str(), std::ifstream::binary);
	int sum = 0;

	if (bmp256.is_open())
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
		delete[] buffer;
	}
	else
	{
		cout << "Can not the BMP file!" << filename << endl;
		return 0;
	}
	bmp256.close();
	return sum / width / height;
}

void WriteBMP256(string filename, int width, int height, int offset, int** img, vector <char> header)
{
	ofstream bmp256(filename.c_str(), std::ofstream::binary);
	if (bmp256)
	{
		bmp256.write((char*)& header[0], offset * sizeof(char));
		char* buffer = new char[width * height];

		int i = 0;
		for (int row = height - 1; row >= 0; row--) {
			for (int col = 0; col < width; col++) {
				buffer[i] = (char)img[row][col];
				i++;
			}
		}

		bmp256.write(buffer, width * height);
		delete[] buffer;

	}
	bmp256.close();
}

void normalize(int width, int height, int** image, int** nml, int normal, int keep)
{
	int mul = 0x01 << keep;
	for (int row = 0; row < height; row++)
	{
		for (int col = 0; col < width; col++) nml[row][col] = image[row][col] * mul / normal;
	}
}

void integralAll(int** image, int** integral, int width, int height)
{
	// int** rowsum = (int**)malloc(sizeof(int) * height);
	// for (int i = 0; i < height; i++) rowsum[i] = (int*)malloc(sizeof(int) * width);
	int** rowsum = new int* [height];
	for (int i = 0; i < height; i++) rowsum[i] = new int[width];

	for (int row = 0; row < height; row++)
	{
		for (int col = 0; col < width; col++)
		{
			rowsum[row][col] = (col - 1 >= 0) ? rowsum[row][col - 1] + image[row][col] : image[row][col];
			integral[row][col] = (row - 1 >= 0) ? integral[row - 1][col] + rowsum[row][col] : rowsum[row][col];
		}
	}
}

void makeBox(TableList box, int** img)
{
	if (box.id == 1)
	{
		for (int row = box.xs; row <= box.xe; row++)
		{
			img[box.ys][row] = 255;
			img[box.ye][row] = 255;
		}

		for (int col = box.ys; col <= box.ye; col++)
		{
			img[col][box.xs] = 255;
			img[col][box.xe] = 255;
		}
	}
}

int Type3(TableList box, int** integral)
{
	TableList negbox, posbox;
	negbox.ys = box.ys; negbox.xs = box.xs;                         negbox.ye = box.ye; negbox.xe = box.xs + (box.xe - box.xs) / 2;
	posbox.ys = box.ys; posbox.xs = box.xs + (box.xe - box.xs) / 2; posbox.ye = box.ye; posbox.xe = box.xe;
	int negRigion = integral[negbox.ys][negbox.xs] + integral[negbox.ye][negbox.xe] - integral[negbox.ys][negbox.xe] - integral[negbox.ye][negbox.xs];
	int posRigion = integral[posbox.ys][posbox.xs] + integral[posbox.ye][posbox.xe] - integral[posbox.ys][posbox.xe] - integral[posbox.ye][posbox.xs];
	return posRigion - negRigion;
}

int Type2(TableList box, int** integral)
{
	TableList negbox1, negbox2, posbox1, posbox2;
	negbox1.ys = box.ys;                         negbox1.xs = box.xs;                         negbox1.ye = box.ys + (box.ye - box.ys) / 2; negbox1.xe = box.xs + (box.xe - box.xs) / 2;
	negbox2.ys = box.ys + (box.ye - box.ys) / 2; negbox2.xs = box.xs + (box.xe - box.xs) / 2; negbox2.ye = box.ye;                         negbox2.xe = box.xe;
	posbox1.ys = box.ys;                         posbox1.xs = box.xs + (box.xe - box.xs) / 2; posbox1.ye = box.ys + (box.ye - box.ys) / 2; posbox1.xe = box.xe;
	posbox2.ys = box.ys + (box.ye - box.ys) / 2; posbox2.xs = box.xs;                         posbox2.ye = box.ye;                         posbox2.xe = box.xs + (box.xe - box.xs) / 2;
	int negRigion1 = integral[negbox1.ys][negbox1.xs] + integral[negbox1.ye][negbox1.xe] - integral[negbox1.ys][negbox1.xe] - integral[negbox1.ye][negbox1.xs];
	int negRigion2 = integral[negbox2.ys][negbox2.xs] + integral[negbox2.ye][negbox2.xe] - integral[negbox2.ys][negbox2.xe] - integral[negbox2.ye][negbox2.xs];
	int posRigion1 = integral[posbox1.ys][posbox1.xs] + integral[posbox1.ye][posbox1.xe] - integral[posbox1.ys][posbox1.xe] - integral[posbox1.ye][posbox1.xs];
	int posRigion2 = integral[posbox2.ys][posbox2.xs] + integral[posbox2.ye][posbox2.xe] - integral[posbox2.ys][posbox2.xe] - integral[posbox2.ye][posbox2.xs];
	return posRigion1 + posRigion2 - negRigion1 - negRigion2;
}

int Type1(TableList box, int** integral)
{
	TableList negbox1, posbox1, posbox2;
	posbox1.ys = box.ys;                             posbox1.xs = box.xs; posbox1.ye = box.ys + (box.ye - box.ys) / 3;                          posbox1.xe = box.xe;
	posbox2.ys = box.ys + (box.ye - box.ys) * 2 / 3; posbox2.xs = box.xs; posbox2.ye = box.ye;                                                  posbox2.xe = box.xe;
	negbox1.ys = box.ys + (box.ye - box.ys) / 3;                          negbox1.xs = box.xs; negbox1.ye = box.ys + (box.ye - box.ys) * 2 / 3; negbox1.xe = box.xe;
	int negRigion1 = integral[negbox1.ys][negbox1.xs] + integral[negbox1.ye][negbox1.xe] - integral[negbox1.ys][negbox1.xe] - integral[negbox1.ye][negbox1.xs];
	int posRigion1 = integral[posbox1.ys][posbox1.xs] + integral[posbox1.ye][posbox1.xe] - integral[posbox1.ys][posbox1.xe] - integral[posbox1.ye][posbox1.xs];
	int posRigion2 = integral[posbox2.ys][posbox2.xs] + integral[posbox2.ye][posbox2.xe] - integral[posbox2.ys][posbox2.xe] - integral[posbox2.ye][posbox2.xs];
	return  posRigion1 + posRigion2 - negRigion1;
}

int Type0(TableList box, int** integral)
{
	TableList negbox1, posbox1, posbox2, posbox3, posbox4;
	negbox1.ys = box.ys + (box.ye - box.ys) / 4;     negbox1.xs = box.xs + (box.xe - box.xs) / 4;     negbox1.ye = box.ys + (box.ye - box.ys) * 3 / 4; negbox1.xe = box.xs + (box.xe - box.xs) * 3 / 4;
	posbox1.ys = box.ys;                             posbox1.xs = box.xs;                             posbox1.ye = box.ye;                             posbox1.xe = box.xs + (box.xe - box.xs) / 4;
	posbox2.ys = box.ys;                             posbox2.xs = box.xs + (box.xe - box.xs) * 3 / 4; posbox2.ye = box.ye;                             posbox2.xe = box.xe;
	posbox3.ys = box.ys;                             posbox3.xs = box.xs + (box.xe - box.xs) / 4;     posbox3.ye = box.ys + (box.ye - box.ys) / 4;     posbox3.xe = box.xs + (box.xe - box.xs) * 3 / 4;
	posbox4.ys = box.ys + (box.ye - box.ys) * 3 / 4; posbox4.xs = box.xs + (box.xe - box.xs) / 4;     posbox4.ye = box.ye;                             posbox4.xe = box.xs + (box.xe - box.xs) * 3 / 4;
	int negRigion1 = integral[negbox1.ys][negbox1.xs] + integral[negbox1.ye][negbox1.xe] - integral[negbox1.ys][negbox1.xe] - integral[negbox1.ye][negbox1.xs];
	int posRigion1 = integral[posbox1.ys][posbox1.xs] + integral[posbox1.ye][posbox1.xe] - integral[posbox1.ys][posbox1.xe] - integral[posbox1.ye][posbox1.xs];
	int posRigion2 = integral[posbox2.ys][posbox2.xs] + integral[posbox2.ye][posbox2.xe] - integral[posbox2.ys][posbox2.xe] - integral[posbox2.ye][posbox2.xs];
	int posRigion3 = integral[posbox3.ys][posbox3.xs] + integral[posbox3.ye][posbox3.xe] - integral[posbox3.ys][posbox3.xe] - integral[posbox3.ye][posbox3.xs];
	int posRigion4 = integral[posbox4.ys][posbox4.xs] + integral[posbox4.ye][posbox4.xe] - integral[posbox4.ys][posbox4.xe] - integral[posbox4.ye][posbox4.xs];
	return posRigion1 + posRigion2 + posRigion3 + posRigion4 - negRigion1;
}

void GetTrainTable(string filename, vector <FeatureList>& FeatureTable)
{
	ifstream TrainTable(filename.c_str());
	string line;
	int index;
	FeatureList Table_tmp;
	int i = 0, cnt = 0;
	if (TrainTable.is_open())
	{
		getline(TrainTable, line);
		while (!TrainTable.eof())
		{
			TrainTable >> Table_tmp.box.id >> index >> Table_tmp.box.xs >> Table_tmp.box.ys
				>> Table_tmp.box.xe >> Table_tmp.box.ye >> Table_tmp.thp >> Table_tmp.thn >> Table_tmp.alpha;
			// !(Table_tmp.thp == 0 && Table_tmp.thn == 0)
			if (Table_tmp.thp != 0 || Table_tmp.thn != 0)
			{
				FeatureTable.push_back(Table_tmp);
				i++;
			}
			cnt++;
		}
	}
	cout << "Out of " << cnt << " from TrainTable " << i << " were inserted into FeatureTable" << endl;
	TrainTable.close();
}

float sumAlphas(vector <FeatureList> FeatureTable)
{
	int n = FeatureTable.size();
	float total = 0.0;
	for (int i = 0; i < n; i++)	total += FeatureTable[i].alpha;
	return total;
}

bool xsSmall(FeatureList i, FeatureList j) { return (i.box.xs < j.box.xs); }
bool ysSmall(FeatureList i, FeatureList j) { return (i.box.ys < j.box.ys); }
bool xeLarge(FeatureList i, FeatureList j) { return (i.box.xe > j.box.xe); }
bool yeLarge(FeatureList i, FeatureList j) { return (i.box.ye > j.box.ye); }

vector <TableList> DetectEye(ofstream& TableOut, vector <FeatureList> FeatureTable, int** integral, int imgsizeW, float halfAlpha)
{
	int fv;
	int type;
	FeatureList min_it;
	vector <FeatureList> Left, Right;
	TableList box;
	vector <TableList> final_box, empty;
	vector <int> skip;
	// float passedAlpha = 0.0;
	bool cascadeFailed = false;
	int numFeatures = FeatureTable.size();
	for (int i = 0; i < numFeatures; i++)
	{
		// if (passedAlpha >= halfAlpha)	break;
		int FeatureType = FeatureTable[i].box.id % 4;
		if (FeatureType == 0)		fv = Type0(FeatureTable[i].box, integral); 
		else if (FeatureType == 1)	fv = Type1(FeatureTable[i].box, integral);
		else if (FeatureType == 2)	fv = Type2(FeatureTable[i].box, integral);
		else						fv = Type3(FeatureTable[i].box, integral);

		// Only push_back if feature passes if-conditional (threshold) statements
		if (!((fv < FeatureTable[i].thp && fv > FeatureTable[i].thn) ||
			(fv >= 0 && FeatureTable[i].thp == 0) ||
			(fv <= 0 && FeatureTable[i].thn == 0)))
		{
			// passedAlpha += FeatureTable[i].alpha;
			if (FeatureTable[i].box.xs < (3 + imgsizeW / 4) && FeatureTable[i].box.xe < (6 + imgsizeW / 2)) Left.push_back(FeatureTable[i]);
			else Right.push_back(FeatureTable[i]);
			if (TableOut.is_open())
			{
				TableOut << FeatureTable[i].box.id << "	" << FeatureType
					<< "	" << FeatureTable[i].box.xs
					<< "	" << FeatureTable[i].box.ys
					<< "	" << FeatureTable[i].box.xe
					<< "	" << FeatureTable[i].box.ye
					<< "	" << FeatureTable[i].thp
					<< "	" << FeatureTable[i].thn
					<< "	" << FeatureTable[i].alpha << endl;
			}
		}
		// else
		// {
		// 	// Since this is already sorted by largest to smallest alpha,
		// 	// this is effectively a cascade already; Just need to quit
		// 	// early for failing at a step.
		// 	cascadeFailed = true;
		// 	break;
		// }
		
		// vector<int> lookFor;
		// lookFor.reserve(3);
		// if (FeatureTable[i].box.id % 4 == 0) 
		// {
		// 	type = 0;
		// 	fv = Type0(FeatureTable[i].box, integral); 
		// 	lookFor.push_back(FeatureTable[i].box.id+1);
		// 	lookFor.push_back(FeatureTable[i].box.id+2);
		// 	lookFor.push_back(FeatureTable[i].box.id+3);
		// }
		// else if (FeatureTable[i].box.id % 4 == 1) 
		// {
		// 	type = 1;
		// 	fv = Type1(FeatureTable[i].box, integral);
		// 	lookFor.push_back(FeatureTable[i].box.id-1);
		// 	lookFor.push_back(FeatureTable[i].box.id+1);
		// 	lookFor.push_back(FeatureTable[i].box.id+2);
		// }
		// else if (FeatureTable[i].box.id % 4 == 2) 
		// {
		// 	type = 2;
		// 	fv = Type2(FeatureTable[i].box, integral);
		// 	lookFor.push_back(FeatureTable[i].box.id-2);
		// 	lookFor.push_back(FeatureTable[i].box.id-1);
		// 	lookFor.push_back(FeatureTable[i].box.id+1);
		// }
		// else 
		// {
		// 	type = 3;
		// 	fv = Type3(FeatureTable[i].box, integral);
		// 	lookFor.push_back(FeatureTable[i].box.id-3);
		// 	lookFor.push_back(FeatureTable[i].box.id-2);
		// 	lookFor.push_back(FeatureTable[i].box.id-1);
		// }
	
	// 	// True then current i is in vector skip and should be skipped!
	// 	bool skipping = find(skip.begin(), skip.end(), i) != skip.end();
	// 	if (!((fv < FeatureTable[i].thp && fv > FeatureTable[i].thn) ||
	// 		(fv >= 0 && FeatureTable[i].thp == 0) ||
	// 		(fv <= 0 && FeatureTable[i].thn == 0)) && !skipping)
	// 	{
	// 		int lookCount = 0;
	// 		vector<int> ids = {i};
	// 		// Find the FeatureTable indexes of the other 3 if they exists and if they also passes threshold
	// 		for (int j = 0; j < 4; j++)
	// 		{
	// 			if (j != type)
	// 			{
	// 				for(int k = 0; k < FeatureTable.size(); k++)
	// 				{
	// 					if(FeatureTable[k].box.id == lookFor[lookCount])
	// 					{	
	// 						int fv0;
	// 						if (FeatureTable[k].box.id % 4 == 0)	fv0 = Type0(FeatureTable[k].box, integral); 
	// 						else if (FeatureTable[k].box.id % 4 == 0)	fv0 = Type1(FeatureTable[k].box, integral);
	// 						else if (FeatureTable[k].box.id % 4 == 0)	fv0 = Type2(FeatureTable[k].box, integral);
	// 						else fv0 = Type3(FeatureTable[k].box, integral);
	// 						// Only push_back if feature passes if-conditional (threshold) statements
	// 						if (!((fv0 < FeatureTable[i].thp && fv0 > FeatureTable[i].thn) ||
	// 							(fv0 >= 0 && FeatureTable[i].thp == 0) ||
	// 							(fv0 <= 0 && FeatureTable[i].thn == 0)))
	// 						{
	// 							ids.push_back(k);
	// 							skip.push_back(k);
	// 						}
	// 					}
	// 				}
	// 				lookCount++;
	// 			}
	// 		}
	// 		// 0 for 1/4, 1 for 2/4, 2 for 3/4, 3 for 4/4 features in the quad that passed the threshold.
	// 		if (ids.size() > 0)
	// 		{
	// 			for(int m = 0; m < ids.size(); m++)
	// 			{
	// 				if (FeatureTable[ids[m]].box.xs < (3 + imgsizeW / 4) && FeatureTable[ids[m]].box.xe < (6 + imgsizeW / 2)) Left.push_back(FeatureTable[ids[m]]);
	// 				else Right.push_back(FeatureTable[ids[m]]);
	// 				if (TableOut.is_open())
	// 				{
	// 					TableOut << FeatureTable[ids[m]].box.id << "	" << FeatureTable[ids[m]].box.id % 4
	// 						<< "	" << FeatureTable[ids[m]].box.xs
	// 						<< "	" << FeatureTable[ids[m]].box.ys
	// 						<< "	" << FeatureTable[ids[m]].box.xe
	// 						<< "	" << FeatureTable[ids[m]].box.ye
	// 						<< "	" << FeatureTable[ids[m]].thp
	// 						<< "	" << FeatureTable[ids[m]].thn
	// 						<< "	" << FeatureTable[ids[m]].alpha << endl;
	// 				}
	// 			}
	// 		}

	// 		/*for(int m = 0; m < ids.size(); m++)
	// 		{
	// 			if (FeatureTable[ids[m]].box.xs < (3 + imgsizeW / 4) && FeatureTable[ids[m]].box.xe < (6 + imgsizeW / 2)) Left.push_back(FeatureTable[ids[m]]);
	// 			else Right.push_back(FeatureTable[ids[m]]);
	// 			if (TableOut.is_open())
	// 			{
	// 				TableOut << FeatureTable[ids[m]].box.id << "	" << FeatureTable[ids[m]].box.id % 4
	// 					<< "	" << FeatureTable[ids[m]].box.xs
	// 					<< "	" << FeatureTable[ids[m]].box.ys
	// 					<< "	" << FeatureTable[ids[m]].box.xe
	// 					<< "	" << FeatureTable[ids[m]].box.ye
	// 					<< "	" << FeatureTable[ids[m]].thp
	// 					<< "	" << FeatureTable[ids[m]].thn
	// 					<< "	" << FeatureTable[ids[m]].alpha << endl;
	// 			}
	// 		}*/
	// 	}
	}

	// if ((double)passedAlpha >= (double)(0.5 * totalAlpha))
	{
		if (Left.size() == 0) box.id = 0;
		else
		{
			box.id = 1;
			std::sort(Left.begin(), Left.end(), xsSmall); box.xs = Left[0].box.xs;
			std::sort(Left.begin(), Left.end(), ysSmall); box.ys = Left[0].box.ys;
			std::sort(Left.begin(), Left.end(), xeLarge); box.xe = Left[0].box.xe;
			std::sort(Left.begin(), Left.end(), yeLarge); box.ye = Left[0].box.ye;
		}
		final_box.push_back(box);

		if (Right.size() == 0) box.id = 0;
		else
		{
			box.id = 1;
			std::sort(Right.begin(), Right.end(), xsSmall); box.xs = Right[0].box.xs;
			std::sort(Right.begin(), Right.end(), ysSmall); box.ys = Right[0].box.ys;
			std::sort(Right.begin(), Right.end(), xeLarge); box.xe = Right[0].box.xe;
			std::sort(Right.begin(), Right.end(), yeLarge); box.ye = Right[0].box.ye;
		}
		final_box.push_back(box);
	}

	// if (cascadeFailed)
	// if (passedAlpha < halfAlpha)
	// {
	// 	final_box[0].id = 0;
	// 	final_box[1].id = 0;
	// }
	
	return final_box;
}

bool alphaLargeSmall(FeatureList i, FeatureList j) { return (i.alpha > j.alpha); }
bool idEqual(FeatureList i, FeatureList j) { return (i.box.id == j.box.id); }

void WriteTrainTable(string infilename, string outfilename)
{
	ifstream inTable(infilename.c_str());
	ofstream outTable(outfilename.c_str());
	string line;
	int index;
	FeatureList Table_tmp;
	vector <FeatureList> FeatureTable;

	if (inTable.is_open())
	{
		outTable << "INDEX TYPE XSTART YSTART XEND YEND THRESHOLD_P THRESHOLD_N Alpha" << endl;

		while (!inTable.eof())
		{
			inTable >> Table_tmp.box.id >> index >> Table_tmp.box.xs >> Table_tmp.box.ys
				>> Table_tmp.box.xe >> Table_tmp.box.ye >> Table_tmp.thp >> Table_tmp.thn >> Table_tmp.alpha;

			vector<FeatureList> searchlist; searchlist.push_back(Table_tmp);
			vector<FeatureList>::iterator flag = std::search(FeatureTable.begin(), FeatureTable.end(), searchlist.begin(), searchlist.end(), idEqual);
			if (flag  == FeatureTable.end() || FeatureTable.size() == 0) FeatureTable.push_back(Table_tmp);
		}
	}

	std::sort(FeatureTable.begin(), FeatureTable.end(), alphaLargeSmall);

	for (int i = 0; i < FeatureTable.size();i++)
	{
		outTable << FeatureTable[i].box.id << "	" << FeatureTable[i].box.id % 4
			<< "	" << FeatureTable[i].box.xs
			<< "	" << FeatureTable[i].box.ys
			<< "	" << FeatureTable[i].box.xe
			<< "	" << FeatureTable[i].box.ye
			<< "	" << FeatureTable[i].thp
			<< "	" << FeatureTable[i].thn
			<< "	" << FeatureTable[i].alpha << endl;
	}

	inTable.close();
	outTable.close();
}

int main(int argc, char** argv)
{
	auto start = high_resolution_clock::now();

	string WorkDrive = "C:";
	int img_start = 20, img_end = 29;
	int img_cnt = img_end - img_start + 1;
	
	int imgsizeW, imgsizeH, offset = 0;

	if (argc != 4)
	{
		std::cout << "WorkDrive Start End" << endl;
		return 0;
	}
	else
	{
		try
		{
			WorkDrive = argv[1];
			img_start = atoi(argv[2]); img_end = atoi(argv[3]); img_cnt = img_end - img_start + 1;
		}
		catch(const std::exception& e)
		{
			std::cerr << e.what() << endl;
		}
	}
	
	string bmpsource = WorkDrive + WorkFolder + ImageFolder + "trainimg_0.bmp";

	vector <char> header;
	ReadBMP256Size(bmpsource, &imgsizeW, &imgsizeH, &offset, header);
	
	int** img = new int* [imgsizeH]; for (int i = 0; i < imgsizeH; i++) img[i] = new int[sizeof(int) * imgsizeW];
	int** nml = new int* [imgsizeH]; for (int i = 0; i < imgsizeH; i++) nml[i] = new int[sizeof(int) * imgsizeW];
	int** integral = new int* [imgsizeH]; for (int i = 0; i < imgsizeH; i++) integral[i] = new int[sizeof(int) * imgsizeW];

	string TrainTablefilename = WorkDrive + WorkFolder + "TrainTable_mem.txt";
	vector <FeatureList> FeatureTable;
	GetTrainTable(TrainTablefilename, FeatureTable);

	string TableOutfilename = WorkDrive + WorkFolder + "TableOut.txt";
	ofstream TableOut(TableOutfilename.c_str());

	float totalAlpha = sumAlphas(FeatureTable);
	float halfAlpha = 0.5 * totalAlpha;

	for (int k = 0; k < img_cnt; k++)
	{
		std::cout << k << " ";
		stringstream ss;
		ss << k;
		string bmpsource = WorkDrive + WorkFolder + ImageFolder + "trainimg_" + ss.str() + ".bmp";
		string bmpoutput = WorkDrive + WorkFolder + ImgOutFolder + "detect_" + ss.str() + ".bmp";

		int avg = ReadBMP256(bmpsource, imgsizeW, imgsizeH, offset, img);
		normalize(imgsizeW, imgsizeH, img, nml, avg, 8); //keep 8 bits of floating point
		integralAll(nml, integral, imgsizeW, imgsizeH);

		vector <TableList> final_box = DetectEye(TableOut, FeatureTable, integral, imgsizeW, halfAlpha);
		// vector <TableList> final_box = DetectEye(TableOut, FeatureTable, integral, imgsizeW, 0.0);

		makeBox(final_box[0], img); makeBox(final_box[1], img);
		WriteBMP256(bmpoutput, imgsizeW, imgsizeH, offset, img, header);
	}

	string SortOutfilename = WorkDrive + WorkFolder + "SortOut.txt";
	WriteTrainTable(TableOutfilename, SortOutfilename);
	for (int i = 0; i < imgsizeH; i++) delete[] img[i]; delete[] img;
	for (int i = 0; i < imgsizeH; i++) delete[] nml[i]; delete[] nml;
	for (int i = 0; i < imgsizeH; i++) delete[] integral[i]; delete[] integral;

	auto end = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(end-start);
    long conv = duration.count();
    std::cout << "\n--- " << format_duration(conv) << " ---" << endl;

	return 0;
}
