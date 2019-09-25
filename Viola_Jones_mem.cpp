#include <iostream>
#include <fstream>
#include <string>
#include <cstdlib>
#include <vector>
#include <sstream>
#include <cmath>
#include <algorithm>    
#include <chrono>
// #include <opencv2/opencv.hpp>
//#include <array> 
//#include <iomanip>      // std::setw

//#include <Magick++.h>

using namespace std;
using namespace std::chrono;
//using namespace Magick;
#define DEBUG_IMGFEATURE

/*
img_0.bmp etc for testing, trainimg_0.bmp etc for training.
/Desktop/CPP/Viola_Jones/ for Linux
/Users/infin/OneDrive/NeuronBasic/Viola-Jones-Eye-Detection/ for Windows
*/

string WorkFolder = "/Users/infin/OneDrive/NeuronBasic/Viola-Jones-Eye-Detection/";
string SourceEyeTableFilename = "eye_point_data.txt";

string TrainFolder = "trainimg/";
// string OutputFolder = "trained/";
string ImgPrefix = "trainimg_";

string format_duration(long dur) {
	microseconds milsec(dur);
	auto ms = duration_cast<milliseconds>(milsec);		milsec -= duration_cast<microseconds>(ms);
	auto secs = duration_cast<seconds>(ms);				ms -= duration_cast<milliseconds>(secs);
	auto mins = duration_cast<minutes>(secs);			secs -= duration_cast<seconds>(mins);
	auto hour = duration_cast<hours>(mins);				mins -= duration_cast<minutes>(hour);

	stringstream ss;
	// ss << hour.count() << " hours " << mins.count() << " minutes " << secs.count() << " seconds " <<
	// 		ms.count() << " milliseconds " << milsec.count() << " microseconds";
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

struct FeatureValue {
//	TableList box;
	int fv;
	int hit;
};

struct FeatureThreshold {
	TableList box;
	int thp;
	int thn;
};

struct TrainOut {
	int minidx;
	double beta;

};


void GetEyeList(string filename, vector <TableList> &LeftTable, vector <TableList> &RightTable, vector <TableList> &LeftMinMax, vector <TableList> &RightMinMax, int imgsizeW) {
	
	ifstream ifs(filename.c_str());
	string line;
	TableList Table_tmp;
	vector <TableList> Table;
	vector <int> vtmp;
	int tmp, num = 0, type = 0;
	int value[5];

	if (ifs.is_open())
	{
		while (getline(ifs, line))
		{
			int lineLength = line.length();
			for (int i = 0; i < lineLength-1; i++)
			{
				char c = line[i];
				if (line[i] == '[')
				{
					type = 0;
				}
				else if (line[i] > 0x2F && line[i] < 0x3A)
				{
					tmp = line[i] - 0x30;
					num = num * 10 + tmp;
				}
				else if (line[i] == ',')
				{
					value[type] = num;
					type++;
					num = 0;
				}
				else if (line[i] == ']')
				{
					value[type] = num;
					Table_tmp.id = value[0];
					Table_tmp.xs = value[1];
					Table_tmp.ys = value[2];
					Table_tmp.xe = value[3];
					Table_tmp.ye = value[4];
#ifdef DEBUG
					if (Table_tmp.id == 69)
					{
						int aa = 1;
					}
#endif
					if (Table_tmp.xs < (3 + imgsizeW / 4) && Table_tmp.xe < (6 + imgsizeW / 2)) LeftTable.push_back(Table_tmp);
					else RightTable.push_back(Table_tmp);

					num = 0;
					type = 0;
				} 
			}
		}
	}
	ifs.close();

	TableList minid_tmp, maxid_tmp;;

	vtmp.clear();
	for (int i = 0; i < LeftTable.size(); i++) vtmp.push_back(LeftTable[i].xs);
	Table_tmp.xs = *std::min_element(vtmp.begin(), vtmp.end());
	value[1] = *std::max_element(vtmp.begin(), vtmp.end());
	minid_tmp.xs = LeftTable[std::min_element(vtmp.begin(), vtmp.end()) - vtmp.begin()].id;
	maxid_tmp.xs = LeftTable[std::max_element(vtmp.begin(), vtmp.end()) - vtmp.begin()].id;
	vtmp.clear();
	for (int i = 0; i < LeftTable.size(); i++) vtmp.push_back(LeftTable[i].ys);
	Table_tmp.ys = *std::min_element(vtmp.begin(), vtmp.end());
	value[2] = *std::max_element(vtmp.begin(), vtmp.end());	
	minid_tmp.ys = LeftTable[std::min_element(vtmp.begin(), vtmp.end()) - vtmp.begin()].id;
	maxid_tmp.ys = LeftTable[std::max_element(vtmp.begin(), vtmp.end()) - vtmp.begin()].id;
	vtmp.clear();
	for (int i = 0; i < LeftTable.size(); i++) vtmp.push_back(LeftTable[i].xe);
	Table_tmp.xe = *std::min_element(vtmp.begin(), vtmp.end());
	value[3] = *std::max_element(vtmp.begin(), vtmp.end());
	minid_tmp.xe = LeftTable[std::min_element(vtmp.begin(), vtmp.end()) - vtmp.begin()].id;
	maxid_tmp.xe = LeftTable[std::max_element(vtmp.begin(), vtmp.end()) - vtmp.begin()].id;
	vtmp.clear();
	for (int i = 0; i < LeftTable.size(); i++) vtmp.push_back(LeftTable[i].ye);
	Table_tmp.ye = *std::min_element(vtmp.begin(), vtmp.end());
	value[4] = *std::max_element(vtmp.begin(), vtmp.end());
	minid_tmp.ye = LeftTable[std::min_element(vtmp.begin(), vtmp.end()) - vtmp.begin()].id;
	maxid_tmp.ye = LeftTable[std::max_element(vtmp.begin(), vtmp.end()) - vtmp.begin()].id;
	LeftMinMax.push_back(Table_tmp);
	Table_tmp.xs = value[1];
	Table_tmp.ys = value[2];
	Table_tmp.xe = value[3];
	Table_tmp.ye = value[4];
	LeftMinMax.push_back(Table_tmp);
	LeftMinMax.push_back(minid_tmp);
	LeftMinMax.push_back(maxid_tmp);

	vtmp.clear();
	for (int i = 0; i < RightTable.size(); i++) vtmp.push_back(RightTable[i].xs);
	Table_tmp.xs = *std::min_element(vtmp.begin(), vtmp.end());
	value[1] = *std::max_element(vtmp.begin(), vtmp.end());
	minid_tmp.xs = RightTable[std::min_element(vtmp.begin(), vtmp.end()) - vtmp.begin()].id;
	maxid_tmp.xs = RightTable[std::max_element(vtmp.begin(), vtmp.end()) - vtmp.begin()].id;
	vtmp.clear();
	for (int i = 0; i < RightTable.size(); i++) vtmp.push_back(RightTable[i].ys);
	Table_tmp.ys = *std::min_element(vtmp.begin(), vtmp.end());
	value[2] = *std::max_element(vtmp.begin(), vtmp.end());
	minid_tmp.ys = RightTable[std::min_element(vtmp.begin(), vtmp.end()) - vtmp.begin()].id;
	maxid_tmp.ys = RightTable[std::max_element(vtmp.begin(), vtmp.end()) - vtmp.begin()].id;
	vtmp.clear();
	for (int i = 0; i < RightTable.size(); i++) vtmp.push_back(RightTable[i].xe);
	Table_tmp.xe = *std::min_element(vtmp.begin(), vtmp.end());
	value[3] = *std::max_element(vtmp.begin(), vtmp.end());
	minid_tmp.xe = RightTable[std::min_element(vtmp.begin(), vtmp.end()) - vtmp.begin()].id;
	maxid_tmp.xe = RightTable[std::max_element(vtmp.begin(), vtmp.end()) - vtmp.begin()].id;
	vtmp.clear();
	for (int i = 0; i < RightTable.size(); i++) vtmp.push_back(RightTable[i].ye);
	Table_tmp.ye = *std::min_element(vtmp.begin(), vtmp.end());
	value[4] = *std::max_element(vtmp.begin(), vtmp.end());
	minid_tmp.ye = RightTable[std::min_element(vtmp.begin(), vtmp.end()) - vtmp.begin()].id;
	maxid_tmp.ye = RightTable[std::max_element(vtmp.begin(), vtmp.end()) - vtmp.begin()].id;
	RightMinMax.push_back(Table_tmp);
	Table_tmp.xs = value[1];
	Table_tmp.ys = value[2];
	Table_tmp.xe = value[3];
	Table_tmp.ye = value[4];
	RightMinMax.push_back(Table_tmp);
	RightMinMax.push_back(minid_tmp);
	RightMinMax.push_back(maxid_tmp);

#ifdef DEBUG
	string EyeListFilename = WorkDrive + WorkFolder + "EyeList.txt";
	ofstream EyeList(EyeListFilename.c_str());
	EyeList << "[";
	int Li = 0, Ri = 0;
	while(Li < LeftTable.size() && Ri < RightTable.size())
	{
		if (LeftTable[Li].id <= RightTable[Ri].id)
		{
			EyeList << "[" << (LeftTable[Li].id) << ", "
				<< (LeftTable[Li].xs) << ", " << (LeftTable[Li].ys) << ", "
				<< (LeftTable[Li].xe) << ", " << (LeftTable[Li].ye) << "], ";
			Li++;
		}
		else
		{
			EyeList << "[" << (RightTable[Ri].id) << ", "
				<< (RightTable[Ri].xs) << ", " << (RightTable[Ri].ys) << ", "
				<< (RightTable[Ri].xe) << ", " << (RightTable[Ri].ye) << "], ";
			Ri++;
		}
	}
	for (int i = Li; i < LeftTable.size();i++)
	{
		EyeList << "[" << (LeftTable[i].id) << ", "
			<< (LeftTable[i].xs) << ", " << (LeftTable[i].ys) << ", "
			<< (LeftTable[i].xe) << ", " << (LeftTable[i].ye) << "]";
	}
	for (int i = Ri; i < RightTable.size();i++)
	{
		EyeList << "[" << (RightTable[i].id) << ", "
			<< (RightTable[i].xs) << ", " << (RightTable[i].ys) << ", "
			<< (RightTable[i].xe) << ", " << (RightTable[i].ye) << "]";
	}
	EyeList << "]";
	EyeList.close();
#endif

	cout << "[Left Eye]" << endl;
	cout << "Image " << LeftMinMax[2].xs << " : xs_min = " << LeftMinMax[0].xs 
		<< ", Image " << LeftMinMax[2].ys << " : ys_min = " << LeftMinMax[0].ys
		<< ", Image " << LeftMinMax[2].xe << " : xe_min = " << LeftMinMax[0].xe
		<< ", Image " << LeftMinMax[2].ye << " : ye_min = " << LeftMinMax[0].ye << endl;

	cout << "Image " << LeftMinMax[3].xs << " : xs_max = " << LeftMinMax[1].xs
		<< ", Image " << LeftMinMax[3].ys << " : ys_max = " << LeftMinMax[1].ys
		<< ", Image " << LeftMinMax[3].xe << " : xe_max = " << LeftMinMax[1].xe
		<< ", Image " << LeftMinMax[3].ye << " : ye_max = " << LeftMinMax[1].ye << endl;

	cout << "[Right Eye]" << endl;
	cout << "Image " << RightMinMax[2].xs << " : xs_min = " << RightMinMax[0].xs
		<< ", Image " << RightMinMax[2].ys << " : ys_min = " << RightMinMax[0].ys
		<< ", Image " << RightMinMax[2].xe << " : xe_min = " << RightMinMax[0].xe
		<< ", Image " << RightMinMax[2].ye << " : ye_min = " << RightMinMax[0].ye << endl;

	cout << "Image " << RightMinMax[3].xs << " : xs_max = " << RightMinMax[1].xs
		<< ", Image " << RightMinMax[3].ys << " : ys_max = " << RightMinMax[1].ys
		<< ", Image " << RightMinMax[3].xe << " : xe_max = " << RightMinMax[1].xe
		<< ", Image " << RightMinMax[3].ye << " : ye_max = " << RightMinMax[1].ye << endl;
}

void BuildFeatureLoc(vector <TableList>& FeatureLoc, vector <TableList> LeftMinMax, vector <TableList> RightMinMax, int FaceWidth)
{
	TableList Table_tmp;
	bool same = false;
	int id = 0;

	//	int xblock = 2, yblock = 2, offset = FaceWidth / 16;
	//	int xblock = 2, yblock = 2, offset = 0; //
	int xblock = 1, yblock = 1, offset = 0, xsdiv = 1, ysdiv = 1, xediv = 1, yediv = 1; //
	int ysmin = LeftMinMax[0].ys - offset; if (ysmin < 0) ysmin = 0;
	int xsmin = LeftMinMax[0].xs - offset; if (xsmin < 0) xsmin = 0;
	int yemin = LeftMinMax[0].ye - offset; if (yemin < 0) yemin = 0;
	int xemin = LeftMinMax[0].xe - offset; if (xemin < 0) xemin = 0;	
	int ysmax = LeftMinMax[1].ys + offset; if (ysmax > 31) ysmax = 31;
	int xsmax = LeftMinMax[1].xs + offset; if (xsmax > 31) xsmax = 31;
	int yemax = LeftMinMax[1].ye + offset; if (yemax > 31) yemax = 31;
	int xemax = LeftMinMax[1].xe + offset; if (xemax > 31) xemax = 31;
	for (int ys = ysmin; ys <= ysmax; ys++)
	{
		for (int xs = xsmin; xs <= xsmax; xs++)
		{
			for (int ye = yemin; ye <= yemax; ye++)
			{
				for (int xe = xemin; xe <= xemax; xe++)
				{
					for (int ysdiv = 0; ysdiv < yblock; ysdiv++)
					{
						for (int xsdiv = 0; xsdiv < xblock; xsdiv++)
						{
							for (int yediv = 0; yediv < yblock; yediv++)
							{
								for (int xediv = 0; xediv < xblock; xediv++)
								{
									Table_tmp.id = id;
									Table_tmp.xs = xs;
									Table_tmp.ys = ys;
									Table_tmp.xe = xe;
									Table_tmp.ye = ye;

								/*	Table_tmp.xs = xs + (xe - xs) * xsdiv / xblock;
									Table_tmp.ys = ys + (ye - ys) * ysdiv / yblock;
									Table_tmp.xe = xe - (xe - xs) * xediv / xblock;
									Table_tmp.ye = ye - (ye - ys) * yediv / yblock;*/
									
									FeatureLoc.push_back(Table_tmp);
									id++;
								}
							}
						} 
					}
				}
			}
		}
	}

	ysmin = RightMinMax[0].ys - offset; if (ysmin < 0) ysmin = 0;
	xsmin = RightMinMax[0].xs - offset; if (xsmin < 0) xsmin = 0;
	yemin = RightMinMax[0].ye - offset; if (yemin < 0) yemin = 0;
	xemin = RightMinMax[0].xe - offset; if (xemin < 0) xemin = 0;
	ysmax = RightMinMax[1].ys + offset; if (ysmax > 31) ysmax = 31;
	xsmax = RightMinMax[1].xs + offset; if (xsmax > 31) xsmax = 31;
	yemax = RightMinMax[1].ye + offset; if (yemax > 31) yemax = 31;
	xemax = RightMinMax[1].xe + offset; if (xemax > 31) xemax = 31;

	for (int ys = ysmin; ys <= ysmax; ys++)
	{
		for (int xs = xsmin; xs <= xsmax; xs++)
		{
			for (int ye = yemin; ye <= yemax; ye++)
			{
				for (int xe = xemin; xe <= xemax; xe++)
				{
					for (int ysdiv = 0; ysdiv < yblock; ysdiv++)
					{
						for (int xsdiv = 0; xsdiv < xblock; xsdiv++)
						{
							for (int yediv = 0; yediv < yblock; yediv++)
							{
								for (int xediv = 0; xediv < xblock; xediv++)
								{
									Table_tmp.id = id;
									Table_tmp.xs = xs;
									Table_tmp.ys = ys;
									Table_tmp.xe = xe;
									Table_tmp.ye = ye;

								/*	Table_tmp.xs = xs + (xe - xs) * xsdiv / xblock;
									Table_tmp.ys = ys + (ye - ys) * ysdiv / yblock;
									Table_tmp.xe = xe - (xe - xs) * xediv / xblock;
									Table_tmp.ye = ye - (ye - ys) * yediv / yblock;*/
									
									FeatureLoc.push_back(Table_tmp);
									id++;
								}
							}
						}
					}
				}
			}
		}
	}
#ifdef DEBUG
	string FeatureLocFilename = WorkDrive + WorkFolder + "FeatureLocation.txt";
	ofstream FeatureLocation(FeatureLocFilename.c_str());
	for (int i = 0; i < FeatureLoc.size();i++)
	{
		FeatureLocation << FeatureLoc[i].id << "	" << FeatureLoc[i].xs << "	" 
			<< FeatureLoc[i].ys << "	" << FeatureLoc[i].xe << "	" << FeatureLoc[i].ye << endl;
	}
#endif
}

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

void normalize(int width, int height, int** image, int normal, int keep)
{
	int mul = 0x01 << keep;
	for (int row = 0; row < height; row++)
	{
		for (int col = 0; col < width; col++) image[row][col] = image[row][col] * mul / normal;
	}
}

void integralAll(int** image, int** integral, int width, int height)
{

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
	for (int i = 0; i < height; i++) delete[] rowsum[i]; delete[] rowsum;

}

/*
unsigned int getIntegral(int** image, int cornerWidth, int cornerHeight)
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
}
*/

int Type3(TableList box, int** integral)
{
	TableList negbox, posbox;
	negbox.ys = box.ys; negbox.xs = box.xs;                         negbox.ye = box.ye; negbox.xe = box.xs + (box.xe - box.xs) / 2;
	posbox.ys = box.ys; posbox.xs = box.xs + (box.xe - box.xs) / 2; posbox.ye = box.ye; posbox.xe = box.xe;
	int negRigion = integral[negbox.ys][negbox.xs] + integral[negbox.ye][negbox.xe] - integral[negbox.ys][negbox.xe] - integral[negbox.ye][negbox.xs];
	int posRigion = integral[posbox.ys][posbox.xs] + integral[posbox.ye][posbox.xe] - integral[posbox.ys][posbox.xe] - integral[posbox.ye][posbox.xs];
	return posRigion - negRigion;
}

/*int Type1(TableList box, int** integral)
{
	TableList negbox, posbox;
	negbox.ys = box.ys;                         negbox.xs = box.xs; negbox.ye = box.ys + (box.ye - box.ys) /2; negbox.xe = box.xe;
	posbox.ys = box.ys + (box.ye - box.ys) / 2; posbox.xs = box.xs; posbox.ye = box.ye;                        posbox.xe = box.xe;
	int negRigion = integral[negbox.ys][negbox.xs] + integral[negbox.ye][negbox.xe] - integral[negbox.ys][negbox.xe] - integral[negbox.ye][negbox.xs];
	int posRigion = integral[posbox.ys][posbox.xs] + integral[posbox.ye][posbox.xe] - integral[posbox.ys][posbox.xe] - integral[posbox.ye][posbox.xs];
	return posRigion - negRigion;
}*/

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


void AllImageFeature(vector <TableList> FeatureLoc, int** img, int** integral, 
	                 FeatureValue** FeatureImageAll, int k, int img_cnt,
	                 vector <TableList> LeftTable, vector <TableList> RightTable)
{
	FeatureValue vtmp;
	TableList box_tmp;
	int j;
	for (int i = 0; i < FeatureLoc.size() * 4; i++)
	{
		box_tmp = FeatureLoc[i / 4];
		box_tmp.id = k;
		if (i % 4 == 0) vtmp.fv = Type0(FeatureLoc[i / 4], integral);
		else if (i % 4 == 1) vtmp.fv = Type1(FeatureLoc[i / 4], integral);
		else if (i % 4 == 2) vtmp.fv = Type2(FeatureLoc[i / 4], integral);
		else vtmp.fv = Type3(FeatureLoc[i / 4], integral);

		vtmp.hit = 0;
		j = 0;
		while (j < img_cnt && j < LeftTable.size() && LeftTable[j].id != k) j++;
		if (j < LeftTable.size() && LeftTable[j].id == k &&
			box_tmp.xs >= LeftTable[j].xs &&
			box_tmp.ys >= LeftTable[j].ys &&
			box_tmp.xe <= LeftTable[j].xe &&
			box_tmp.ye <= LeftTable[j].ye) vtmp.hit = 1;

		j = 0;
		while (j < img_cnt && j < RightTable.size() && RightTable[j].id != k) j++;
		if (j < RightTable.size() && RightTable[j].id == k && vtmp.hit == 0 &&
			box_tmp.xs >= RightTable[j].xs &&
			box_tmp.ys >= RightTable[j].ys &&
			box_tmp.xe <= RightTable[j].xe &&
			box_tmp.ye <= RightTable[j].ye) vtmp.hit = 1;

		FeatureImageAll[i][k] = vtmp;
	}
}

void BuildFeatureThreshold(FeatureValue** FeatureImageAll, vector <TableList> FeatureLoc,
	                       vector <FeatureThreshold>& ThresholdTable, int img_cnt)
{
	vector <FeatureValue> ImageFeature;
	FeatureValue Feature_tmp;
	vector <int> thp, thn;
	int FeatureLen = FeatureLoc.size() * 4;
	cout << "Finding Plus Threshold : ";
	for (int fid = 0; fid < FeatureLen; fid++)
	{
		if (fid % 100000 == 0 || fid == FeatureLen - 1) cout << fid << " ";
//		cout << fid << " ";
		int totalhit = 0;
		int totalsample = 0;
		int save = 0;
		double gini = 0;
		double minGini = 1;
		for (int img = 0; img < img_cnt; img++)
		{
			if (FeatureImageAll[fid][img].fv > 0)
			{
				totalhit += FeatureImageAll[fid][img].hit;
				totalsample++;
			}
		}
		for (int img = 0; img < img_cnt; img++)
		{
			double EN = 0;
			double EP = 0;
			if (FeatureImageAll[fid][img].fv > 0)
			{
				int th = FeatureImageAll[fid][img].fv;
				int errorNeg = 0;
				int errorPos = 0;
				for (int idx = 0; idx < img_cnt; idx++)
				{
					if (FeatureImageAll[fid][idx].fv > 0)
					{
						if (FeatureImageAll[fid][idx].fv < th &&
							FeatureImageAll[fid][idx].hit == 1)
							errorPos++;
						else if (FeatureImageAll[fid][idx].fv >= th &&
								 FeatureImageAll[fid][idx].hit == 0)
							errorNeg++;
					}
				}
				if (totalhit == totalsample) EN = 1;
				else EN = ((double)errorNeg) / ((double)(totalsample - totalhit));
				if (totalhit == 0) EP = 1;
				else EP = ((double)errorPos) / ((double) totalhit);
				gini = EN * EN + EP * EP;
				if (gini < minGini)
				{
					save = th;
					minGini = gini;
				}
			}
		}
		thp.push_back(save);
	}

	cout << endl << "Finding Minus Threshold : ";


	for (int fid = 0; fid < FeatureLen; fid++)
	{
		if (fid % 100000 == 0 || fid == FeatureLen - 1) cout << fid << " ";
//		cout << fid << " ";
		int totalhit = 0;
		int totalsample = 0;
		int save = 0;
		double gini = 0;
		double minGini = 1;		
		for (int img = 0; img < img_cnt; img++)
		{
			if (FeatureImageAll[fid][img].fv <= 0)
			{
				totalhit += FeatureImageAll[fid][img].hit;
				totalsample++;
			}
		}

		for (int img = 0; img < img_cnt; img++)
		{
			double EN = 0;
			double EP = 0;
			if (FeatureImageAll[fid][img].fv <= 0)
			{
				int th = FeatureImageAll[fid][img].fv;
				int errorNeg = 0;
				int errorPos = 0;
				for (int idx = 0; idx < img_cnt; idx++)
				{
					if (FeatureImageAll[fid][idx].fv <= 0)
					{
						if (FeatureImageAll[fid][idx].fv > th &&
							FeatureImageAll[fid][idx].hit == 1)
							errorPos++;
						else if (FeatureImageAll[fid][idx].fv <= th &&
							FeatureImageAll[fid][idx].hit == 0)
							errorNeg++;
					}
				}
				if (totalhit == totalsample) EN = 1;
				else EN = ((double)errorNeg) / ((double)(totalsample - totalhit));
				if (totalhit == 0) EP = 1;
				else EP = ((double)errorPos) / ((double)totalhit);
				gini = EN * EN + EP * EP;
				if (gini < minGini)
				{
					save = th;
					minGini = gini;
				}
			}
		}
		thn.push_back(save);
	}

	cout << endl;

	FeatureThreshold vtmp;
	for (int fid = 0; fid < FeatureLen; fid++)
	{
		vtmp.box = FeatureLoc[fid / 4];
		vtmp.box.id = fid;
		vtmp.thp = thp[fid];
		vtmp.thn = thn[fid];
		ThresholdTable.push_back(vtmp);
	}
}

void BuildThresholdHit(FeatureValue** FeatureImageAll, vector <int>& ThresholdHit, vector <int>& FeatureHit, 
	                   vector <FeatureThreshold> ThresholdTable, int img_cnt)
{
	for (int fid = 0; fid < ThresholdTable.size(); fid++)
	{
		for (int img = 0; img < img_cnt; img++)
		{
			FeatureHit.push_back(FeatureImageAll[fid][img].hit);
			// if ((FeatureImageAll[fid][img].fv < ThresholdTable[fid].thp && FeatureImageAll[fid][img].fv > ThresholdTable[fid].thn) ||
			// 	(FeatureImageAll[fid][img].fv >= 0 && ThresholdTable[fid].thp == 0) || (FeatureImageAll[fid][img].fv <= 0 && ThresholdTable[fid].thn == 0))
			if ((FeatureImageAll[fid][img].fv < ThresholdTable[fid].thp && FeatureImageAll[fid][img].fv > ThresholdTable[fid].thn) || 
				(ThresholdTable[fid].thp == 0 && ThresholdTable[fid].thn == 0))
				ThresholdHit.push_back(0);
			else
				ThresholdHit.push_back(1);
		}
	}
}


void initWeights(vector <int> FeatureHit, double** weights, int FeatureLen, int img_cnt)
{

	for (int fid = 0; fid < FeatureLen; fid++)
	{
		int negcnt = 0;
		int poscnt = 0;
		for (int img = 0; img < img_cnt; img++)
		{
			if (FeatureHit[fid * img_cnt + img] == 0) negcnt++;
			else poscnt++;
		}
		for (int img = 0; img < img_cnt; img++)
		{
			if (FeatureHit[fid * img_cnt + img] == 0)
			{
				if (poscnt == 0) weights[fid][img] = 1.0 / double(negcnt);
				else weights[fid][img] = 1.0 / double(negcnt * 2);
			}
			else
			{
				if (negcnt == 0) weights[fid][img] = 1.0 / double(poscnt);
				else weights[fid][img] = 1.0 / double(poscnt * 2);
			}
		}
	}
}

void normWeights(double** weights, int FeatureLen, int img_cnt)
{
	double weightSum = 0;
	for (int f = 0; f < FeatureLen; f++)
	{
		for (int i = 0; i < img_cnt; i++) weightSum += weights[f][i];
	}
	if (weightSum == 0) weightSum = 1.0;
	for (int f = 0; f < FeatureLen; f++)
	{
		for (int i = 0; i < img_cnt; i++) weights[f][i] /= weightSum;
	}
}

TrainOut train(std::ofstream& TableOut, double** weights, vector <int> ThresholdHit,
	vector <int> FeatureHit, vector <FeatureThreshold> ThresholdTable, vector <int> MinIndex, int img_cnt)
{

	vector <double> featureError;
	double alpha;
	int FeatureLen = ThresholdTable.size();
	for (int fid = 0; fid < FeatureLen; fid++)
	{
		double sum = 0;
		for (int img = 0; img < img_cnt; img++)
		{
			if (ThresholdHit[fid * img_cnt + img] != FeatureHit[fid * img_cnt + img])
			{
				sum += weights[fid][img];
			}
		}
		for (int i = 0; i < MinIndex.size(); i++)
		{
			if (fid == MinIndex[i]) sum = 100.0;
		}
		featureError.push_back(sum);
	}

	int index = std::min_element(featureError.begin(), featureError.end()) - featureError.begin();
	double minError = *std::min_element(featureError.begin(), featureError.end());

	double beta = minError / (1.0 - minError);
	if (beta == 0) alpha = 50;
	else alpha = log(1 / beta);

	if (TableOut.is_open() )
	{
		TableOut << index << "	" << index % 4
			<< "	" << ThresholdTable[index].box.xs
			<< "	" << ThresholdTable[index].box.ys
			<< "	" << ThresholdTable[index].box.xe
			<< "	" << ThresholdTable[index].box.ye
			<< "	" << ThresholdTable[index].thp
			<< "	" << ThresholdTable[index].thn
			<< "	" << alpha << endl;
	}

/*
	TableOut << setw(5) << index << setw(4) << index % 4
		<< setw(6) << ThresholdTable[index].box.xs
		<< setw(6) << ThresholdTable[index].box.ys
		<< setw(4) << ThresholdTable[index].box.xe
		<< setw(4) << ThresholdTable[index].box.ye
		<< setw(12) << ThresholdTable[index].thp
		<< setw(12) << ThresholdTable[index].thn
		<< setw(12) << alpha;
*/
/*
	fprintf(TableOut, "%5d %4d %6d %6d %4d %4d %12d %12d %12f\n",
		index, index % 4, ThresholdTable[index].box.xs, ThresholdTable[index].box.ys,
		ThresholdTable[index].box.xe, ThresholdTable[index].box.ye,
		ThresholdTable[index].thp, ThresholdTable[index].thn, alpha);
		*/
	TrainOut tmp;
	tmp.beta = beta;
	tmp.minidx = index;
	return tmp;
}

void updateWeights(double** weights, vector <int> ThresholdHit, vector <int> FeatureHit, int FeatureLen, int img_cnt, int minIndex, double beta)
{

	for (int img = 0; img < img_cnt; img++)
	{
		if (FeatureHit[minIndex * img_cnt + img] == ThresholdHit[minIndex * img_cnt + img]) weights[minIndex][img] *= beta;
	}
}

/*
Filter the all features in FeatureImageAll hits 1/0 (P/N) so that P/total# >= percentage 
	and N/total# >= percentage remain only before passing to building thresholds
*/
void FilterFeatures(string WorkDrive, FeatureValue** FeatureImageAll, vector <TableList> &FeatureLoc, int img_cnt, int percentage) 
{
	double cutoff = percentage/100.0;
	string statFileName = WorkDrive + WorkFolder + "FeatureStat.txt";
	ofstream FeatureStats(statFileName.c_str());
	FeatureStats << "FeatureIndex Pos Neg" << endl;
	string goodStatFileName = WorkDrive + WorkFolder + "FeatureGood.txt";
	ofstream goodStats(goodStatFileName.c_str());
	goodStats << "FeatureIndex Pos Neg" << endl;
	vector <int> NewFeatureImageAllIndex;
	vector <TableList> NewFeatureLoc;
	// Note: FeatureValues (fv) are different for each type BUT
	// feature's hit/miss is only based on FeatureLocation THUS
	// all features at same location share same hit/miss!!!
	for (int i = 0; i < FeatureLoc.size() * 4; i += 4)
	{
		// @At this feature location...
		int pos = 0, neg = 0;
		for (int j = 0; j < img_cnt; j++)
		{
			if (FeatureImageAll[i][j].hit == 1)	pos++;
			else neg++;
		}
		FeatureStats << i << " " << (double)pos/img_cnt << " " << (double)neg/img_cnt << endl;
		if ((double)pos/img_cnt >= cutoff && (double)neg/img_cnt >= cutoff)
		{
			goodStats << i << " " << pos << " " << neg << endl;
			NewFeatureImageAllIndex.push_back(i);
		}	
	}
	goodStats.close();
	FeatureStats.close();
	FeatureValue** NewFeatureImageAll = new FeatureValue* [NewFeatureImageAllIndex.size() * 4];
	for (int i = 0; i < NewFeatureImageAllIndex.size() * 4; i++) NewFeatureImageAll[i] = new FeatureValue [img_cnt];
	int newIdx = 0;
	for (int i = 0; i < NewFeatureImageAllIndex.size(); i++)
	{
		int oldIdx = NewFeatureImageAllIndex[i];
		NewFeatureLoc.push_back(FeatureLoc[oldIdx / 4]);
		for (int type = 0; type < 4; type++)
		{
			for (int j = 0; j < img_cnt; j++)
			{
				NewFeatureImageAll[newIdx][j] = FeatureImageAll[oldIdx+type][j];
			}
			newIdx++;
		}
	}
	FeatureImageAll = NewFeatureImageAll;
	cout << "Old FeatureLoc size: " << FeatureLoc.size() << endl;
	FeatureLoc.clear();
	FeatureLoc = NewFeatureLoc;
	cout << "New FeatureLoc size: " << FeatureLoc.size() << endl;
}

int main(int argc, char** argv)
{
	auto start = high_resolution_clock::now();

	string WorkDrive = "C:";

	int img_start = 20, img_end = 29;
	int img_cnt = img_end - img_start + 1;


	if (argc != 4)
	{
		cout << "WorkDrive Start End" << endl;
		return 0;
	}
	else
	{
		WorkDrive = argv[1];
		img_start = atoi(argv[2]); img_end = atoi(argv[3]); img_cnt = img_end - img_start + 1;
	}
#ifdef DEBUG
	string ImageFilename = WorkDrive + WorkFolder + "Image.txt";
	ofstream Image(ImageFilename.c_str());
	string NormalFilename = WorkDrive + WorkFolder + "Normal.txt";
	ofstream Normal(NormalFilename.c_str());
	string IntegralFilename = WorkDrive + WorkFolder + "Integral.txt";
	ofstream Integral(IntegralFilename.c_str());
#endif


	int imgsizeW, imgsizeH, offset, length;
	string bmpsource = WorkDrive + WorkFolder + TrainFolder + ImgPrefix + "1.bmp";

	vector <char> header;
	ReadBMP256Size(bmpsource, &imgsizeW, &imgsizeH, &offset, header);

	int** img = new int* [imgsizeH]; for (int i = 0; i < imgsizeH; i++) img[i] = new int[sizeof(int) * imgsizeW];
	int** integral = new int* [imgsizeH]; for (int i = 0; i < imgsizeH; i++) integral[i] = new int[sizeof(int) * imgsizeW];

	string SourceEyeTable = WorkDrive + WorkFolder + SourceEyeTableFilename;
	vector <TableList> LeftTable, RightTable, LeftMinMax, RightMinMax;
	GetEyeList(SourceEyeTable, LeftTable, RightTable, LeftMinMax, RightMinMax, imgsizeW);

	vector <TableList> FeatureLoc;
	BuildFeatureLoc(FeatureLoc, LeftMinMax, RightMinMax, imgsizeW);
	LeftMinMax.clear(); RightMinMax.clear();

	FeatureValue** FeatureImageAll = new FeatureValue* [FeatureLoc.size() * 4];
	for (int i = 0; i < FeatureLoc.size() * 4; i++) FeatureImageAll[i] = new FeatureValue [img_cnt];

	cout << "Calculate image : ";

	for (int k = img_start; k <= img_end; k++)
	{
		stringstream ss;
		ss << k;
		string bmpsource = WorkDrive + WorkFolder + TrainFolder + ImgPrefix + ss.str() + ".bmp";

		int avg = ReadBMP256(bmpsource, imgsizeW, imgsizeH, offset, img);

#ifdef DEBUG
			cout << "average of image " << k << " = " << avg << endl;
			for (int row = 0; row < imgsizeH; row++)
			{
				for (int col = 0; col < imgsizeW; col++)
				{
					Image << img[row][col] << "	";
				}
				Image << endl;
			}
#endif
		normalize(imgsizeW, imgsizeH, img, avg, 8); //keep 8 bits of floating point

		integralAll(img, integral, imgsizeW, imgsizeH);

#ifdef DEBUG
			for (int row = 0; row < imgsizeH; row++)
			{
				for (int col = 0; col < imgsizeW; col++)
				{
					Normal << img[row][col] << "	";
					Integral << integral[row][col] << "	";
				}
				Normal << endl;
				Integral << endl;
			}
#endif
		AllImageFeature(FeatureLoc, img, integral, FeatureImageAll, k, img_cnt, LeftTable, RightTable);

		cout << k << " ";

#ifdef DEBUG
			string FeaturetableFilename = WorkDrive + WorkFolder + "Image" + to_string(k) + "Feature.txt";
			ofstream FeatureTable(FeaturetableFilename.c_str());
			for (int i = 0; i < ImageFeature.size();i++)
				FeatureTable << ImageFeature[i].box.id << "	" << ImageFeature[i].box.xs << "	"
				<< ImageFeature[i].box.ys << "	" << ImageFeature[i].box.xe << "	"
				<< ImageFeature[i].box.ye << "	" << ImageFeature[i].fv << "	" << ImageFeature[i].hit << endl;
			FeatureTable.close();
#endif
	}
	cout << endl;
	for (int i = 0; i < imgsizeH; i++) delete[] img[i]; delete[] img;
	for (int i = 0; i < imgsizeH; i++) delete[] integral[i]; delete[] integral;
	LeftTable.clear(); RightTable.clear();
	
#ifdef DEBUG
	Image.close();
	Normal.close();
	Integral.close();
#endif
	int percentage = 30;
	cout << "Filtering Features @" << percentage << "% (Pre-Processing)......." << endl;
	FilterFeatures(WorkDrive, FeatureImageAll, FeatureLoc, img_cnt, percentage);


	cout << "Finding Threshold......." << endl;

	vector <FeatureThreshold> ThresholdTable;
	BuildFeatureThreshold(FeatureImageAll, FeatureLoc, ThresholdTable, img_cnt);
	FeatureLoc.clear();

#ifdef DEBUG
		string ThresholdTableFilename = WorkDrive + WorkFolder + "ThresholdList.txt";
		ofstream ThresholdList(ThresholdTableFilename.c_str());
		for (int i = 0; i < ThresholdTable.size();i++)
		{
			ThresholdList << ThresholdTable[i].box.id << "	" << ThresholdTable[i].box.xs << "	"
				<< ThresholdTable[i].box.ys << "	" << ThresholdTable[i].box.xe << "	"
				<< ThresholdTable[i].box.ye << "	" << ThresholdTable[i].thp << "	" << ThresholdTable[i].thn << endl;
		}
		ThresholdList.close();
#endif
	int FeatureLen = ThresholdTable.size();
	vector <int> ThresholdHit, FeatureHit;
	BuildThresholdHit(FeatureImageAll, ThresholdHit, FeatureHit, ThresholdTable, img_cnt);
	
	for (int i = 0; i < FeatureLen; i++) delete[] FeatureImageAll[i]; delete[] FeatureImageAll;

	cout << "Release FeatureImage......." << endl;

	double** weights = new double* [FeatureLen]; for (int i = 0; i < FeatureLen; i++) weights[i] = new double[img_cnt];

	cout << "Create Weight......." << endl;



#ifdef DEBUG
		string ThresholdHitFilename = WorkDrive + WorkFolder + "ThresholdHitList.txt";
		ofstream ThresholdHitList(ThresholdHitFilename.c_str());
		for (int fid = 0; fid < ThresholdTable.size(); fid++)
		{
			for (int imgid = 0; imgid < img_cnt; imgid++)
			{
				ThresholdHitList << ThresholdHit[fid * img_cnt + imgid] << "	";
			}
			ThresholdHitList << endl;
		}
		ThresholdHitList.close();
#endif
	initWeights(FeatureHit, weights, FeatureLen, img_cnt);

	string OutputTable = WorkDrive + WorkFolder + "TrainTable.txt";
	ofstream TableOut(OutputTable.c_str());

	vector <int> MinIndex;
	TableOut << "INDEX TYPE XSTART YSTART XEND YEND THRESHOLD_P THRESHOLD_N Alpha" << endl;

	cout << " Training on : ";
	for (int i = 0; i < FeatureLen; i++)
	{
		normWeights(weights, FeatureLen, img_cnt);
		TrainOut minidx_beta = train(TableOut, weights, ThresholdHit, FeatureHit, ThresholdTable, MinIndex, img_cnt);
		MinIndex.push_back(minidx_beta.minidx);
		updateWeights(weights, ThresholdHit, FeatureHit, FeatureLen, img_cnt, minidx_beta.minidx, minidx_beta.beta);
		cout << i << " ";
	}
	cout << endl;
	for (int i = 0; i < FeatureLen; i++) delete[] weights[i]; delete[] weights;

	ThresholdTable.clear(); ThresholdHit.clear();

	auto end = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(end-start);
    long conv = duration.count();
    cout << "\n--- " << format_duration(conv) << " ---" << endl;

	return 0;

}

