#include <iostream>
#include <fstream>
#include <string>
#include <cstdlib>
#include <vector>
#include <sstream>
#include <cmath>
#include <algorithm>    
//#include <array> 
//#include <iomanip>      // std::setw

//#include <Magick++.h>

using namespace std;
//using namespace Magick;
#define DEBUG_IMGFEATURE

//string WorkDrive = "C";
string WorkFolder = "/CPP/Viola_Jones/";
string SourceEyeTableFilename = "eye_point_data.txt";

string TrainFolder = "trainimg/";
string OutputFolder = "trained/";
string ImgPrefix = "trainimg_";



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
	int xblock = 1, yblock = 1, offset = 0, xsdiv = 1, ysdiv = 1, xediv = 1, yediv = 1; 
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
//					for (int ysdiv = 0; ysdiv < yblock; ysdiv++)
//					{
//						for (int xsdiv = 0; xsdiv < xblock; xsdiv++)
//						{
//							for (int yediv = 0; yediv < yblock; yediv++)
//							{
//								for (int xediv = 0; xediv < xblock; xediv++)
//								{
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
//								}
//							}
//						} 
//					}
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
//					for (int ysdiv = 0; ysdiv < yblock; ysdiv++)
//					{
//						for (int xsdiv = 0; xsdiv < xblock; xsdiv++)
//						{
//							for (int yediv = 0; yediv < yblock; yediv++)
//							{
//								for (int xediv = 0; xediv < xblock; xediv++)
//								{
									Table_tmp.id = id;
									Table_tmp.xs = xs;
									Table_tmp.ys = ys;
									Table_tmp.xe = xe;
									Table_tmp.ye = ye;

							/*		Table_tmp.xs = xs + (xe - xs) * xsdiv / xblock;
									Table_tmp.ys = ys + (ye - ys) * ysdiv / yblock;
									Table_tmp.xe = xe - (xe - xs) * xediv / xblock;
									Table_tmp.ye = ye - (ye - ys) * yediv / yblock;*/

									FeatureLoc.push_back(Table_tmp);
									id++;
//								}
//							}
//						}
//					}
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
	// int** rowsum = (int**)malloc(sizeof(int) * height);
	// for (int i = 0; i < height; i++) rowsum[i] = (int*)malloc(sizeof(int) * width);
	int** rowsum = new int* [height]; for (int i = 0; i < height; i++) rowsum[i] = new int[width];

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
	                 vector <FeatureValue> &ImageFeature, int k, int img_cnt, 
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

 		ImageFeature.push_back(vtmp);
	}
}

void BuildFeatureImage(string FeatureListFilename, string FeatureImageFilename, vector <TableList> FeatureLoc,
	int img_start, int img_end)
{
	ofstream FeatureImageList(FeatureImageFilename.c_str(), std::ofstream::binary);
	int img_cnt = img_end - img_start + 1;
	int FeatureLen = FeatureLoc.size() * 4;
//	FeatureValue Feature_tmp[10];
	FeatureValue* Feature_tmp = new FeatureValue[img_cnt];
	FeatureValue** FeatureImage = new FeatureValue* [img_cnt];
	for (int i = 0; i < img_cnt; i++) FeatureImage[i] = new FeatureValue[1000];

//	FeatureValue** FeatureImage = (FeatureValue **)malloc(sizeof(FeatureValue) * img_cnt);
//	for (int i = 0; i < img_cnt; i++) FeatureImage[i] = (FeatureValue*)malloc(sizeof(FeatureValue) * 1000);

	int fid, file_size;
	cout << "Process feature : ";

	for (int img = img_start; img <= img_end; img++)
	{
		file_size = 0;
		while (file_size != sizeof(FeatureValue) * FeatureLen)
		{
			stringstream ss;
			ss << img;
			string FeatureListFilename0 = FeatureListFilename + ss.str() + ".bin";
			ifstream FeatureList(FeatureListFilename0.c_str(), std::ifstream::binary);
			FeatureList.seekg(0, ios::end);
			file_size = FeatureList.tellg();
			FeatureList.close();
		}
	}

	for (fid = 0; fid < (FeatureLen/1000)*1000; fid=fid+1000)
	{
		if (fid % 100000 == 0) cout << fid << " ";

		for (int img = img_start; img <= img_end; img++)
		{
			stringstream ss;
			ss << img;
			string FeatureListFilename0 = FeatureListFilename + ss.str() + ".bin";
			ifstream FeatureList(FeatureListFilename0.c_str(), std::ifstream::binary);
			FeatureList.clear();
			FeatureList.seekg(fid * sizeof(FeatureValue), ios::beg);
			FeatureList.read((char*)& FeatureImage[img][0], sizeof(FeatureValue) * 1000);
			FeatureList.close();
		}
		for (int i = 0; i < 1000; i++)
		{
			for (int img = img_start; img <= img_end; img++) Feature_tmp[img] = FeatureImage[img][i];
			FeatureImageList.write((char*)& Feature_tmp[0], sizeof(FeatureValue) * img_cnt);
		}
	}

	int FeatureLen1 = FeatureLen % 1000;
	if (FeatureLen1 > 0)
	{
		cout << FeatureLen << " ";
		for (int img = img_start; img <= img_end; img++)
		{
			stringstream ss;
			ss << img;
			string FeatureListFilename0 = FeatureListFilename + ss.str() + ".bin";
			ifstream FeatureList(FeatureListFilename0.c_str(), std::ifstream::binary);

			FeatureList.clear();
			FeatureList.seekg(fid * sizeof(FeatureValue), ios::beg);
			FeatureList.read((char*)& FeatureImage[img][0], sizeof(FeatureValue) * FeatureLen1);
			FeatureList.close();
		}
		for (int i = 0; i < FeatureLen1; i++)
		{
			for (int img = img_start; img <= img_end; img++) Feature_tmp[img] = FeatureImage[img][i];
			FeatureImageList.write((char*)& Feature_tmp[0], sizeof(FeatureValue) * img_cnt);
		}
	}
	FeatureImageList.close();	  
	for (int i = 0; i < img_cnt; i++) delete[] FeatureImage[i]; delete[] FeatureImage;
	cout << endl;
}

void MergeFeatureImage(string OldFeatureImageFilename, string NewFeatureImageFilename, vector <TableList> FeatureLoc,
	int img_start, int img_end)
{
	ifstream FeatureImage0(OldFeatureImageFilename.c_str(), std::ifstream::binary);
	ifstream FeatureImage1(NewFeatureImageFilename.c_str(), std::ifstream::binary);
	string AllFeatureImageFilename = OldFeatureImageFilename + "All";
	ofstream outFeatureImage(AllFeatureImageFilename.c_str(), std::ofstream::binary);
	FeatureValue Feature_tmp;
	FeatureValue* Feature_tmp0 = new FeatureValue[img_start];
	FeatureValue* Feature_tmp1 = new FeatureValue[img_end - img_start + 1];

	int FeatureLen = FeatureLoc.size() * 4;

	cout << "Merge feature : ";

	for (int fid = 0; fid < FeatureLen; fid++)
	{
		if (fid % 100000 == 0 || fid == FeatureLen - 1) cout << fid << " ";
#ifdef DEBUG
		stringstream ss;
		ss << fid;
		string FeatureidFilename = WorkDrive + WorkFolder + "MFeature" + ss.str() + ".txt";
		ofstream FeatureImage(FeatureidFilename.c_str());
#endif

		FeatureImage0.read((char*)& Feature_tmp0[0], sizeof(FeatureValue) * img_start);
		outFeatureImage.write((char*)& Feature_tmp0[0], sizeof(FeatureValue) * img_start);

#ifdef DEBUG
		for (int img = 0; img < img_start; img++)
		{
			FeatureImage << Feature_tmp0[img].box.id << "	" << Feature_tmp0[img].box.xs << "	" << Feature_tmp0[img].box.ys << "	"
				<< Feature_tmp0[img].box.xe << "	" << Feature_tmp0[img].box.ye << "	" << Feature_tmp0[img].fv << "	" << Feature_tmp0[img].hit << endl;
		}
#endif
		FeatureImage1.read((char*)& Feature_tmp1[0], sizeof(FeatureValue) * (img_end - img_start + 1));
		outFeatureImage.write((char*)& Feature_tmp1[0], sizeof(FeatureValue) * (img_end - img_start + 1));

#ifdef DEBUG
		for (int img = img_start; img <= img_end; img++)
		{
				FeatureImage << Feature_tmp1[img - img_start].box.id << "	" << Feature_tmp1[img - img_start].box.xs << "	" << Feature_tmp1[img - img_start].box.ys << "	"
				<< Feature_tmp1[img - img_start].box.xe << "	" << Feature_tmp1[img - img_start].box.ye << "	" << Feature_tmp1[img - img_start].fv << "	" << Feature_tmp1[img - img_start].hit << endl;
		}
		FeatureImage.close();
#endif
	}
	FeatureImage0.close(); 
	FeatureImage1.close(); 
	outFeatureImage.close();
	delete[] Feature_tmp0;
	delete[] Feature_tmp1;
	cout << endl;
}

void BuildFeatureThreshold(string FeatureImageFilename, vector <TableList> FeatureLoc,
	                       vector <FeatureThreshold>& ThresholdTable, int img_cnt)
{
	ifstream FeatureImageList(FeatureImageFilename.c_str(), std::ifstream::binary);
	FeatureValue* ImageFeature = new FeatureValue[img_cnt * 1000];
	vector <int> thp, thn;
	int FeatureLen = FeatureLoc.size() * 4;
	cout << "Finding Plus Threshold : ";

	int file_size = 0;
	while (file_size != sizeof(FeatureValue) * FeatureLen * img_cnt)
	{
		FeatureImageList.seekg(0, ios::end);
		file_size = FeatureImageList.tellg();
		FeatureImageList.clear();
		FeatureImageList.seekg(0, ios_base::beg);
	}

	for (int fid = 0; fid < FeatureLen; fid++)
	{
		if (fid % 100000 == 0 || fid == FeatureLen - 1) cout << fid << " ";

		if((fid % 1000 == 0) && ((FeatureLen - fid) / 1000 > 0)) FeatureImageList.read((char*)& ImageFeature[0], sizeof(FeatureValue) * img_cnt * 1000);
		else if((fid % 1000 == 0) && ((FeatureLen - fid) / 1000 == 0)) FeatureImageList.read((char*)& ImageFeature[0], sizeof(FeatureValue) * img_cnt * (FeatureLen % 1000));
	
		int totalhit = 0;
		int totalsample = 0;
		int save = 0;
		double gini = 0;
		double minGini = 1;
		for (int img = 0; img < img_cnt; img++)
		{
			if (ImageFeature[(fid % 1000) * img_cnt + img].fv > 0)
			{
				totalhit += ImageFeature[(fid % 1000) * img_cnt + img].hit;
				totalsample++;
			}
		}
		for (int img = 0; img < img_cnt; img++)
		{
			double EN = 0;
			double EP = 0;
			if (ImageFeature[(fid % 1000) * img_cnt + img].fv > 0)
			{
				int th = ImageFeature[(fid % 1000) * img_cnt + img].fv;
				int errorNeg = 0;
				int errorPos = 0;
				for (int idx = 0; idx < img_cnt; idx++)
				{
					if (ImageFeature[(fid % 1000) * img_cnt + idx].fv > 0)
					{
						if (ImageFeature[(fid % 1000) * img_cnt + idx].fv < th &&
							ImageFeature[(fid % 1000) * img_cnt + idx].hit == 1)
							errorPos++;
						else if (ImageFeature[(fid % 1000) * img_cnt + idx].fv >= th &&
								 ImageFeature[(fid % 1000) * img_cnt + idx].hit == 0)
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

	FeatureImageList.clear();
	FeatureImageList.seekg(0, ios::beg);
	cout << endl << "Finding Minus Threshold : ";

	for (int fid = 0; fid < FeatureLen; fid++)
	{
		if(fid % 100000 == 0 || fid == FeatureLen - 1) cout << fid << " ";

//		FeatureImageList.read((char*)& ImageFeature[0], sizeof(FeatureValue) * img_cnt);
		if ((fid % 1000 == 0) && ((FeatureLen - fid) / 1000 > 0)) FeatureImageList.read((char*)& ImageFeature[0], sizeof(FeatureValue) * img_cnt * 1000);
		else if ((fid % 1000 == 0) && ((FeatureLen - fid) / 1000 == 0)) FeatureImageList.read((char*)& ImageFeature[0], sizeof(FeatureValue) * img_cnt * (FeatureLen % 1000));

		int totalhit = 0;
		int totalsample = 0;
		int save = 0;
		double gini = 0;
		double minGini = 1;		
		for (int img = 0; img < img_cnt; img++)
		{
			if (ImageFeature[(fid % 1000) * img_cnt + img].fv <= 0)
			{
				totalhit += ImageFeature[(fid % 1000) * img_cnt + img].hit;
				totalsample++;
			}
		}

		for (int img = 0; img < img_cnt; img++)
		{
			double EN = 0;
			double EP = 0;
			if (ImageFeature[(fid % 1000) * img_cnt + img].fv <= 0)
			{
				int th = ImageFeature[(fid % 1000) * img_cnt + img].fv;
				int errorNeg = 0;
				int errorPos = 0;
				for (int idx = 0; idx < img_cnt; idx++)
				{
					if (ImageFeature[(fid % 1000) * img_cnt + idx].fv <= 0)
					{
						if (ImageFeature[(fid % 1000) * img_cnt + idx].fv > th &&
							ImageFeature[(fid % 1000) * img_cnt + idx].hit == 1)
							errorPos++;
						else if (ImageFeature[(fid % 1000) * img_cnt + idx].fv <= th &&
							ImageFeature[(fid % 1000) * img_cnt + idx].hit == 0)
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
	FeatureImageList.close();
	delete[] ImageFeature;

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

void BuildThresholdHit(string FeatureImageFilename, string ThresholdHitFilename, vector <FeatureThreshold> ThresholdTable, int img_cnt)
{
	cout << endl << "BuildThresholdHit......" << endl;

	ifstream FeatureImageList(FeatureImageFilename.c_str(), std::ifstream::binary);
	ofstream ThresholdHit(ThresholdHitFilename.c_str(), std::ofstream::binary);
	FeatureValue* Feature_tmp = new FeatureValue[img_cnt * 1000];
	int* Hit_tmp = new int[img_cnt * 1000];
	int FeatureLen = ThresholdTable.size();
	int tt=0;
//	int fv[1000 * 10], thp[1000 * 10], thn[1000 * 10], hit[1000 * 10];

	for (int fid = 0; fid < FeatureLen; fid++)
	{
		if ((fid % 1000 == 0) && ((FeatureLen - fid) / 1000 > 0)) FeatureImageList.read((char*)& Feature_tmp[0], sizeof(FeatureValue) * img_cnt * 1000);
		else if ((fid % 1000 == 0) && ((FeatureLen - fid) / 1000 == 0)) FeatureImageList.read((char*)& Feature_tmp[0], sizeof(FeatureValue) * img_cnt * (FeatureLen % 1000));

		for (int img = 0; img < img_cnt; img++)
		{
			if ((Feature_tmp[(fid % 1000) * img_cnt + img].fv < ThresholdTable[fid].thp && Feature_tmp[(fid % 1000) * img_cnt + img].fv > ThresholdTable[fid].thn) ||
				(Feature_tmp[(fid % 1000) * img_cnt + img].fv >= 0 && ThresholdTable[fid].thp == 0) || (Feature_tmp[(fid % 1000) * img_cnt + img].fv <= 0 && ThresholdTable[fid].thn == 0))
				Hit_tmp[(fid % 1000) * img_cnt + img] = 0;
			else
				Hit_tmp[(fid % 1000) * img_cnt + img] = 1;

/*			int i = (fid % 1000) * img_cnt + img;
			fv[i] = Feature_tmp[(fid % 1000) * img_cnt + img].fv;
			thp[i] = ThresholdTable[fid].thp;
			thn[i] = ThresholdTable[fid].thn;
			hit[i] = Hit_tmp[(fid % 1000) * img_cnt + img];*/
		}
		if (((fid + 1) % 1000 == 0) && ((fid + 1) / 1000 > 0))
		{
			ThresholdHit.write((char*)& Hit_tmp[0], sizeof(int) * img_cnt * 1000);
			tt++;
		}
	}
	if (FeatureLen % 1000 > 0)
	{
		ThresholdHit.write((char*)& Hit_tmp[0], sizeof(int) * img_cnt * (FeatureLen % 1000));
		tt++;
	}

	FeatureImageList.close();
	ThresholdHit.close();
	delete[] Feature_tmp;
	delete[] Hit_tmp;

}


void initWeights(string FeatureImageFilename, string Weight0Filename, int FeatureLen, int img_cnt)
{
	cout << endl << "initWeights......" << endl;

	ifstream FeatureImageList(FeatureImageFilename.c_str(), std::ifstream::binary);
	ofstream WeightInit(Weight0Filename.c_str(), std::ofstream::binary);
	FeatureValue* ImageFeature = new FeatureValue[img_cnt * 1000];
	double* Weights = new double[img_cnt * 1000];

	for (int fid = 0; fid < FeatureLen; fid++)
	{
		if ((fid % 1000 == 0) && ((FeatureLen - fid) / 1000 > 0)) FeatureImageList.read((char*)& ImageFeature[0], sizeof(FeatureValue) * img_cnt * 1000);
		else if ((fid % 1000 == 0) && ((FeatureLen - fid) / 1000 == 0)) FeatureImageList.read((char*)& ImageFeature[0], sizeof(FeatureValue) * img_cnt * (FeatureLen % 1000));
		int negcnt = 0;
		int poscnt = 0;
		for (int img = 0; img < img_cnt; img++)
		{
			if (ImageFeature[(fid % 1000) * img_cnt + img].hit == 0) negcnt++;
			else poscnt++;
		}
		for (int img = 0; img < img_cnt; img++)
		{
			if (ImageFeature[(fid % 1000) * img_cnt + img].hit == 0)
			{
				if (poscnt == 0) Weights[(fid % 1000) * img_cnt + img] = (1.0 / double(negcnt));
				else Weights[(fid % 1000) * img_cnt + img] = (1.0 / double(negcnt * 2));
			}
			else
			{
				if (negcnt == 0) Weights[(fid % 1000) * img_cnt + img] = (1.0 / double(poscnt));
				else Weights[(fid % 1000) * img_cnt + img] = (1.0 / double(poscnt * 2));
			}
		}
		if (((fid + 1) % 1000 == 0) && ((fid + 1) / 1000 > 0))
		{
			WeightInit.write((char*)& Weights[0], sizeof(double) * img_cnt * 1000);
			for (int i = 0; i < 1000;i++)
			{
				double ttt = Weights[i];
			}
		}
	}
	WeightInit.write((char*)& Weights[0], sizeof(double) * img_cnt * (FeatureLen % 1000));
	for (int i = 0; i < (FeatureLen % 1000);i++)
	{
		double ttt = Weights[i];
	}
	FeatureImageList.close();
	WeightInit.close();
	delete[] ImageFeature;
	delete[] Weights;

}

void normWeights(string Weight0Filename, string WeightNormalFilename, int FeatureLen, int img_cnt)
{
	
	cout << endl << "normWeights......" << endl;
	ifstream Weight0(Weight0Filename.c_str(), std::ifstream::binary);
	ofstream WeightNormal(WeightNormalFilename.c_str(), std::ofstream::binary);

	double WeightSum = 0;
	double* Weights_tmp = new double[img_cnt * 1000];
	
	int file_size = 0;
	while (file_size != sizeof(double) * FeatureLen * img_cnt)
	{
		Weight0.seekg(0, ios_base::end);
		file_size = Weight0.tellg();
		Weight0.clear();
		Weight0.seekg(0, ios_base::beg);
	}
	
	for (int fid = 0; fid < FeatureLen; fid++)
	{

		if ((fid % 1000 == 0) && ((FeatureLen - fid) / 1000 > 0)) Weight0.read((char*)& Weights_tmp[0], sizeof(double) * img_cnt * 1000);
		else if ((fid % 1000 == 0) && ((FeatureLen - fid) / 1000 == 0)) Weight0.read((char*)& Weights_tmp[0], sizeof(double) * img_cnt * (FeatureLen % 1000));

		for (int i = 0; i < img_cnt; i++) WeightSum += Weights_tmp[(fid % 1000)* img_cnt + i];
	}
	if (WeightSum == 0) WeightSum = 1.0;

	Weight0.seekg(0, ios::beg);
	for (int fid = 0; fid < FeatureLen; fid++)
	{
		if ((fid % 1000 == 0) && ((FeatureLen - fid) / 1000 > 0)) Weight0.read((char*)& Weights_tmp[0], sizeof(double) * img_cnt * 1000);
		else if ((fid % 1000 == 0) && ((FeatureLen - fid) / 1000 == 0)) Weight0.read((char*)& Weights_tmp[0], sizeof(double) * img_cnt * (FeatureLen % 1000));

		for (int i = 0; i < img_cnt; i++) Weights_tmp[(fid % 1000)* img_cnt + i] /= WeightSum;

		if (((fid + 1) % 1000 == 0) && ((fid + 1) / 1000 > 0)) WeightNormal.write((char*)& Weights_tmp[0], sizeof(double) * img_cnt * 1000);
	}
	WeightNormal.write((char*)& Weights_tmp[0], sizeof(double) * img_cnt * (FeatureLen % 1000));

	WeightNormal.close();
	Weight0.close();
	delete[] Weights_tmp;

}

TrainOut train(std::ofstream& TableOut, string WeightNormalFilename, string ThresholdHitFilename,
	string FeatureImageFilename, vector <FeatureThreshold> ThresholdTable, vector <int> MinIndex, int img_cnt)
{
	cout << endl << "train......" << endl;

	ifstream FeatureImageList(FeatureImageFilename.c_str(), std::ifstream::binary);
	ifstream WeightNormal(WeightNormalFilename.c_str(), std::ifstream::binary);
	ifstream ThresholdHit(ThresholdHitFilename.c_str(), std::ifstream::binary);

	FeatureValue* Feature_tmp = new FeatureValue[img_cnt * 1000];
	vector <double> featureError;
	double alpha;
	double* Weights_tmp = new double[img_cnt * 1000];
	int* Hit_tmp = new int[img_cnt * 1000];
	int FeatureLen = ThresholdTable.size();

	int file_size = 0;

	while (file_size != sizeof(double) * FeatureLen * img_cnt)
	{
		WeightNormal.seekg(0, ios_base::end);
		file_size = WeightNormal.tellg();
		WeightNormal.clear();
		WeightNormal.seekg(0, ios_base::beg);
	}

	for (int fid = 0; fid < FeatureLen; fid++)
	{
		double sum = 0;
		if ((fid % 1000 == 0) && ((FeatureLen - fid) / 1000 > 0)) WeightNormal.read((char*)& Weights_tmp[0], sizeof(double) * img_cnt * 1000);
		else if ((fid % 1000 == 0) && ((FeatureLen - fid) / 1000 == 0)) WeightNormal.read((char*)& Weights_tmp[0], sizeof(double) * img_cnt * (FeatureLen % 1000));

		if ((fid % 1000 == 0) && ((FeatureLen - fid) / 1000 > 0)) FeatureImageList.read((char*)& Feature_tmp[0], sizeof(FeatureValue) * img_cnt * 1000);
		else if ((fid % 1000 == 0) && ((FeatureLen - fid) / 1000 == 0)) FeatureImageList.read((char*)& Feature_tmp[0], sizeof(FeatureValue) * img_cnt * (FeatureLen % 1000));

		if ((fid % 1000 == 0) && ((FeatureLen - fid) / 1000 > 0)) ThresholdHit.read((char*)& Hit_tmp[0], sizeof(int) * img_cnt * 1000);
		else if ((fid % 1000 == 0) && ((FeatureLen - fid) / 1000 == 0)) ThresholdHit.read((char*)& Hit_tmp[0], sizeof(int) * img_cnt * (FeatureLen % 1000));

		for (int img = 0; img < img_cnt; img++)
		{
			if (Hit_tmp[(fid % 1000) * img_cnt + img] != Feature_tmp[(fid % 1000) * img_cnt + img].hit) sum += Weights_tmp[(fid % 1000) * img_cnt + img];
		}
		vector <int>::iterator it = find(MinIndex.begin(), MinIndex.end(), fid);
		if (it != MinIndex.end())
		{
			sum = 100.0;
		}

		featureError.push_back(sum);
	}
	FeatureImageList.close();
	WeightNormal.close();
	ThresholdHit.close();
	delete[] Weights_tmp;
	delete[] Feature_tmp;
	delete[] Hit_tmp;

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

	TrainOut tmp;
	tmp.beta = beta;
	tmp.minidx = index;
	return tmp;
}

void updateWeights(string Weight0Filename, string WeightNormalFilename, string ThresholdHitFilename, string FeatureImageFilename, int FeatureLen, int img_cnt, int minIndex, double beta)
{
	cout << endl << "updateWeights......" << endl;

	ifstream FeatureImageList(FeatureImageFilename.c_str(), std::ifstream::binary);
	ifstream WeightNormal(Weight0Filename.c_str(), std::ifstream::binary);
	ofstream Weight0(WeightNormalFilename.c_str(), std::ofstream::binary);
	ifstream ThresholdHit(ThresholdHitFilename.c_str(), std::ifstream::binary);
	FeatureValue *Feature_tmp = new FeatureValue[img_cnt];
	double* Weights_tmp = new double[img_cnt * 1000];
	int* Hit_tmp = new int[img_cnt];

	for (int fid = 0; fid < ((minIndex - 1) / 1000) * 1000; fid = fid + 1000)
	{
		WeightNormal.read((char*)& Weights_tmp[0], sizeof(double) * img_cnt * 1000);
		Weight0.write((char*)& Weights_tmp[0], sizeof(double) * img_cnt * 1000);
	}
	if (minIndex > 0)
	{
		WeightNormal.read((char*)& Weights_tmp[0], sizeof(double) * img_cnt * ((minIndex - 1) % 1000));
		Weight0.write((char*)& Weights_tmp[0], sizeof(double) * img_cnt * ((minIndex - 1) % 1000));
	}

	FeatureImageList.seekg(minIndex * img_cnt * sizeof(FeatureValue), ios::beg);
	FeatureImageList.read((char*)& Feature_tmp[0], sizeof(FeatureValue) * img_cnt);
	WeightNormal.read((char*)& Weights_tmp[0], sizeof(double) * img_cnt);
	ThresholdHit.read((char*)& Hit_tmp[0], sizeof(int) * img_cnt);

	for (int img = 0; img < img_cnt; img++)
	{
		if (Feature_tmp[img].hit == Hit_tmp[img]) Weights_tmp[img] *= beta;
	}

	Weight0.write((char*)& Weights_tmp[0], sizeof(double) * img_cnt);


	WeightNormal.read((char*)& Weights_tmp[0], sizeof(double) * img_cnt * ((FeatureLen - (minIndex + 1)) % 1000));
	Weight0.write((char*)& Weights_tmp[0], sizeof(double) * img_cnt * ((FeatureLen - (minIndex + 1)) % 1000));

	for (int fid = (FeatureLen - (minIndex + 1))/1000; fid == 0; fid--)
	{
		WeightNormal.read((char*)& Weights_tmp[0], sizeof(double) * img_cnt * 1000);
		Weight0.write((char*)& Weights_tmp[0], sizeof(double) * img_cnt * 1000);
	}
	FeatureImageList.close();
	WeightNormal.close();
	Weight0.close();
	ThresholdHit.close();
	delete[] Weights_tmp;
	delete[] Feature_tmp;
	delete[] Hit_tmp;
}

int main(int argc, char** argv)
{
#ifdef DEBUG
	string ImageFilename = WorkDrive + WorkFolder + "Image.txt";
	ofstream Image(ImageFilename.c_str());
	string NormalFilename = WorkDrive + WorkFolder + "Normal.txt";
	ofstream Normal(NormalFilename.c_str());
	string IntegralFilename = WorkDrive + WorkFolder + "Integral.txt";
	ofstream Integral(IntegralFilename.c_str());
#endif
	string WorkDrive = "C";

	int img_start = 20, img_end = 29;
	int img_cnt = img_end - img_start + 1;

	bool BuildImageFeatureOnly = false;
	bool FromBuildFeatureThreshold = false;
	
	if (argc != 4)
	{
		cout << "WorkDrive Start End";
		return 0;
	}
	else
	{
		WorkDrive = argv[1];
		img_start = atoi(argv[2]); img_end = atoi(argv[3]); img_cnt = img_end - img_start + 1;
	}
	
/*
	if (argc != 6)
	{
		cout << "WorkDrive BuildFeatureOnly TrainOnly Start End";
		return 0;
	}
	else
	{
		WorkDrive = argv[1];

		if(atoi(argv[2]) == 1) BuildImageFeatureOnly = true;
		else if (atoi(argv[2]) == 0) BuildImageFeatureOnly = false;
		else 
		{
			cout << "BuildFeatureOnly TrainOnly Start End Total";
			return 0;
		}

		if (atoi(argv[3]) == 1) FromBuildFeatureThreshold = true;
		else if (atoi(argv[3]) == 0) FromBuildFeatureThreshold = false;
		else
		{
			cout << "BuildFeatureOnly TrainOnly Start End Total";
			return 0;
		}

		img_start = atoi(argv[4]); img_end = atoi(argv[5]); img_cnt = img_end - img_start + 1;
		
		for (int i = 0; i < argc; i++) cout << " " << argv[i] << " ";
		cout << endl;
	}
*/
	int imgsizeW, imgsizeH, offset;
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

	string FeatureListFilename, FeatureImageFilename, OldFeatureImageFilename;
	vector <FeatureValue> ImageFeature;

	if (BuildImageFeatureOnly)
	{
		stringstream ss;
		ss << img_start << "_" << img_end;
		FeatureImageFilename = WorkDrive + WorkFolder + "FeatureImage" + ss.str() + ".bin";
		OldFeatureImageFilename = WorkDrive + WorkFolder + "FeatureImage.bin";
		cout << "After this program finished, Please ...." << endl;
		cout << "del " << OldFeatureImageFilename << endl;
		cout << "move " << OldFeatureImageFilename << "All " << OldFeatureImageFilename << endl;
		cout << "Or......" << endl;
		cout << "rm " << OldFeatureImageFilename << endl;
		cout << "mv " << OldFeatureImageFilename << "All " << OldFeatureImageFilename << endl;
	} 
	else
	{
		FeatureImageFilename = WorkDrive + WorkFolder + "FeatureImage.bin";

		if (FromBuildFeatureThreshold)
		{
			cout << "Make sure do below copy file before training................." << endl;
			cout << "copy " << FeatureImageFilename << " " << WorkDrive + WorkFolder + "AllFeatureImage.bin" << endl;
			cout << "Or......" << endl;
			cout << "cp " << FeatureImageFilename << " " << WorkDrive + WorkFolder + "AllFeatureImage.bin" << endl;
		}
	}

	if (!FromBuildFeatureThreshold)
	{
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
			ImageFeature.clear();
			AllImageFeature(FeatureLoc, img, integral, ImageFeature, k, img_cnt, LeftTable, RightTable);
			FeatureListFilename = WorkDrive + WorkFolder + "ImageFeature" + ss.str() + ".bin";
			ofstream FeatureList(FeatureListFilename.c_str(), std::ofstream::binary);
			FeatureList.write((char*)& ImageFeature[0], ImageFeature.size() * sizeof(FeatureValue));
			FeatureList.close();
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
		LeftTable.clear(); RightTable.clear(); ImageFeature.clear();
	}
#ifdef DEBUG
	Image.close();
	Normal.close();
	Integral.close();
#endif
	FeatureListFilename = WorkDrive + WorkFolder + "ImageFeature";
	if (!FromBuildFeatureThreshold)
	{
		BuildFeatureImage(FeatureListFilename, FeatureImageFilename, FeatureLoc, img_start, img_end);

		for (int i = img_start; i <= img_end; i++)
		{
			stringstream ss;
			ss << i;
			string FeatureListFilename0 = FeatureListFilename + ss.str() + ".bin";
			std::remove(FeatureListFilename0.c_str());
		}
	}
	if (BuildImageFeatureOnly) MergeFeatureImage(OldFeatureImageFilename, FeatureImageFilename, FeatureLoc, img_start, img_end);

	cout << "Finding Threshold......." << endl;

	if (!BuildImageFeatureOnly || FromBuildFeatureThreshold)
	{
		if(FromBuildFeatureThreshold) FeatureImageFilename = WorkDrive + WorkFolder + "AllFeatureImage.bin";
		vector <FeatureThreshold> ThresholdTable;
		BuildFeatureThreshold(FeatureImageFilename, FeatureLoc, ThresholdTable, img_cnt);
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

		string ThresholdHitFilename = WorkDrive + WorkFolder + "ThresholdHit.bin";
		BuildThresholdHit(FeatureImageFilename, ThresholdHitFilename, ThresholdTable, img_cnt);

		ifstream ThresholdHit(ThresholdHitFilename.c_str(), std::ifstream::binary);
		int file_size = 0;
		while (file_size != sizeof(int) * FeatureLen * img_cnt)
		{
			ThresholdHit.seekg(0, ios_base::end);
			file_size = ThresholdHit.tellg();
			ThresholdHit.clear();
			ThresholdHit.seekg(0, ios_base::beg);
		}
		ThresholdHit.close();

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
		string Weight0Filename = WorkDrive + WorkFolder + "Weight0.bin";
		string WeightNormalFilename = WorkDrive + WorkFolder + "Weight.bin";
		initWeights(FeatureImageFilename, Weight0Filename, FeatureLen, img_cnt);

		string OutputTable = WorkDrive + WorkFolder + "TrainTable.txt";
		ofstream TableOut(OutputTable.c_str());

		vector <int> MinIndex;

		TableOut << "INDEX TYPE XSTART YSTART XEND YEND THRESHOLD_P THRESHOLD_N Alpha" << endl;

		cout << " Training on : ";
		for (int i = 0; i < FeatureLen; i++)
		{
			normWeights(Weight0Filename, WeightNormalFilename, FeatureLen, img_cnt);
			TrainOut minidx_beta = train(TableOut, WeightNormalFilename, ThresholdHitFilename, FeatureImageFilename, ThresholdTable, MinIndex, img_cnt);
			MinIndex.push_back(minidx_beta.minidx);
			updateWeights(Weight0Filename, WeightNormalFilename, ThresholdHitFilename, FeatureImageFilename, FeatureLen, img_cnt, minidx_beta.minidx, minidx_beta.beta);
			cout << i << " ";
		}
		cout << endl;
		ThresholdTable.clear();
	}
	return 0;

}

