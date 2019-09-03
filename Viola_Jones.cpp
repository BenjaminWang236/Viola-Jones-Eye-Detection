#include <iostream>
#include <fstream>
#include <string>
#include <cstdlib>
#include <vector>
#include <sstream>
#include <cmath>
#include <algorithm>    
#include <array> 
#include <iomanip>      // std::setw

//#include <Magick++.h>

using namespace std;
//using namespace Magick;
#define DEBUG_FEATURE

string WorkFolder = "C:/CPP/Viola_Jones/";
string SourceEyeTableFilename = "eye_point_data.txt";

string TrainFolder = "trainimg/";
string OutputFolder = "trained/";
string ImgPrefix = "trainimg_";

int imgsizeW = 32, imgsizeH = 32, img_cnt = 10;

struct TableList {
	int id;
	int xs;
	int ys;
	int xe;
	int ye;
};

struct FeatureValue {
	TableList box;
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


void GetEyeList(string filename, vector <TableList> &LeftTable, vector <TableList> &RightTable, vector <TableList> &LeftMinMax, vector <TableList> &RightMinMax) {
	
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
	//	auto result = std::minmax_element(vtmp.begin(), vtmp.end());

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
	string EyeListFilename = WorkFolder + "EyeList.txt";
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

	int xblock = 2, yblock = 2, offset = FaceWidth/16;
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
									Table_tmp.xs = xs + (xe - xs) * xsdiv / xblock;
									Table_tmp.ys = ys + (ye - ys) * ysdiv / yblock;
									Table_tmp.xe = xe - (xe - xs) * xediv / xblock;
									Table_tmp.ye = ye - (ye - ys) * yediv / yblock;
									FeatureLoc.push_back(Table_tmp);
									id++;

									/*
									same = false;
									for (int i = 0; i < FeatureLoc.size();i++)
									{
										if (Table_tmp.xs == FeatureLoc[i].xs &&
											Table_tmp.ys == FeatureLoc[i].ys &&
											Table_tmp.xe == FeatureLoc[i].xe &&
											Table_tmp.ye == FeatureLoc[i].ye)
										{
											same = true;
											break;
										}
									}
									if (!same)
									{
										Table_tmp.id = id;
										FeatureLoc.push_back(Table_tmp);
										id++;
									}
									*/
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
									Table_tmp.xs = xs + (xe - xs) * xsdiv / xblock;
									Table_tmp.ys = ys + (ye - ys) * ysdiv / yblock;
									Table_tmp.xe = xe - (xe - xs) * xediv / xblock;
									Table_tmp.ye = ye - (ye - ys) * yediv / yblock;

									FeatureLoc.push_back(Table_tmp);
									id++;

									/*
									same = false;
									for (int i = 0; i < FeatureLoc.size();i++)
									{
										if (Table_tmp.xs == FeatureLoc[i].xs &&
											Table_tmp.ys == FeatureLoc[i].ys &&
											Table_tmp.xe == FeatureLoc[i].xe &&
											Table_tmp.ye == FeatureLoc[i].ye)
										{
											same = true;
											break;
										}
									}
									if (!same)
									{
										Table_tmp.id = id;
										FeatureLoc.push_back(Table_tmp);
										id++;
									}
									*/
								}
							}
						}
					}
				}
			}
		}
	}
#ifdef DEBUG
	string FeatureLocFilename = WorkFolder + "FeatureLocation.txt";
	ofstream FeatureLocation(FeatureLocFilename.c_str());
	for (int i = 0; i < FeatureLoc.size();i++)
	{
		FeatureLocation << FeatureLoc[i].id << "	" << FeatureLoc[i].xs << "	" 
			<< FeatureLoc[i].ys << "	" << FeatureLoc[i].xe << "	" << FeatureLoc[i].ye << endl;
	}
#endif
}

char* ReadBMP256Size(string filename, int* width, int* height, int* offset)
{
	ifstream bmp256(filename.c_str(), std::ifstream::binary);

	if (bmp256)
	{
		char buffer[54];
		bmp256.read(buffer, 54);
		if ((buffer[0] != 'B') || (buffer[1] != 'M'))
		{
			cout << "This is not a BMP file!" << endl;
			bmp256.close();
			return 0;
		}

		*width = *(int*)& buffer[18];
		*height = *(int*)& buffer[22];
		int bits = *(int*)& buffer[28];
		int color = *(int*)& buffer[46];

		if (bits <= 8) *offset = color * 4;
		*offset += 54;

	}
	char* header = new char[*offset];
	bmp256.seekg(0, bmp256.beg);
	bmp256.read(header, *offset);
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

int Type0(TableList box, int** integral)
{
	TableList negbox, posbox;
	negbox.ys = box.ys; negbox.xs = box.xs;                         negbox.ye = box.ye; negbox.xe = box.xs + (box.xe - box.xs) / 2;
	posbox.ys = box.ys; posbox.xs = box.xs + (box.xe - box.xs) / 2; posbox.ye = box.ye; posbox.xe = box.xe;
	int negRigion = integral[negbox.ys][negbox.xs] + integral[negbox.ye][negbox.xe] - integral[negbox.ys][negbox.xe] - integral[negbox.ye][negbox.xs];
	int posRigion = integral[posbox.ys][posbox.xs] + integral[posbox.ye][posbox.xe] - integral[posbox.ys][posbox.xe] - integral[posbox.ye][posbox.xs];
	return posRigion - negRigion;
}

int Type1(TableList box, int** integral)
{
	TableList negbox, posbox;
	negbox.ys = box.ys;                         negbox.xs = box.xs; negbox.ye = box.ys + (box.ye - box.ys) /2; negbox.xe = box.xe;
	posbox.ys = box.ys + (box.ye - box.ys) / 2; posbox.xs = box.xs; posbox.ye = box.ye;                        posbox.xe = box.xe;
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

int Type3(TableList box, int** integral)
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
	                 vector <FeatureValue> &ImageFeature, int k, int imgcnt, 
	                 vector <TableList> LeftTable, vector <TableList> RightTable)
{
	FeatureValue vtmp;
	int j;
	for (int i = 0; i < FeatureLoc.size() * 4; i++)
	{
		vtmp.box.xs = FeatureLoc[i / 4].xs;
		vtmp.box.ys = FeatureLoc[i / 4].ys;
		vtmp.box.xe = FeatureLoc[i / 4].xe;
		vtmp.box.ye = FeatureLoc[i / 4].ye;
		vtmp.box.id = k;
		if (i % 4 == 0) vtmp.fv = Type0(FeatureLoc[i / 4], integral);
		else if (i % 4 == 1) vtmp.fv = Type1(FeatureLoc[i / 4], integral);
		else if (i % 4 == 2) vtmp.fv = Type2(FeatureLoc[i / 4], integral);
		else vtmp.fv = Type3(FeatureLoc[i / 4], integral);

		vtmp.hit = 0;
		j = 0;
		while (j < imgcnt && LeftTable[j].id != k) j++;
		if (LeftTable[j].id == k &&
			vtmp.box.xs >= LeftTable[j].xs &&
			vtmp.box.ys >= LeftTable[j].ys &&
			vtmp.box.xe <= LeftTable[j].xe &&
			vtmp.box.ye <= LeftTable[j].ye) vtmp.hit = 1;

		j = 0;
		while (j < imgcnt && RightTable[j].id != k) j++;
		if (RightTable[j].id == k && vtmp.hit == 0 &&
			vtmp.box.xs >= RightTable[j].xs &&
			vtmp.box.ys >= RightTable[j].ys &&
			vtmp.box.xe <= RightTable[j].xe &&
			vtmp.box.ye <= RightTable[j].ye) vtmp.hit = 1;

		ImageFeature.push_back(vtmp);
	}
}

void BuildFeatureThreshold(string FeatureListFilename, string FeatureImageFilename, vector <TableList> FeatureLoc,
	                       vector <FeatureThreshold>& ThresholdTable, int img_cnt)
{
	ifstream FeatureList(FeatureListFilename.c_str(), std::ifstream::binary);
	ofstream FeatureImageList(FeatureImageFilename.c_str(), std::ofstream::binary);
	vector <FeatureValue> ImageFeature;
	FeatureValue Feature_tmp;
	vector <int> thp, thn;
	int FeatureLen = FeatureLoc.size() * 4;
	int file_id = 0;
	for (int fid = 0; fid < FeatureLen; fid++)
	{
#ifdef DEBUG
		string FeatureidFilename = WorkFolder + "Feature" + to_string(file_id) + ".txt";
		ofstream FeatureImage(FeatureidFilename.c_str());
#endif
		ImageFeature.clear();
		for (int img = 0; img < img_cnt; img++)
		{
			FeatureList.clear();
			FeatureList.seekg(0, ios::beg);
			for (int i = 0; i <= img * FeatureLen + fid; i++) FeatureList.read((char*)& Feature_tmp, sizeof(FeatureValue));
			ImageFeature.push_back(Feature_tmp);
			FeatureImageList.write((char*)& Feature_tmp, sizeof(FeatureValue));

#ifdef DEBUG
			FeatureImage << Feature_tmp.box.id << "	" << Feature_tmp.box.xs << "	" << Feature_tmp.box.ys << "	"
				<< Feature_tmp.box.xe << "	" << Feature_tmp.box.ye << "	" << Feature_tmp.fv << "	" << Feature_tmp.hit  << endl;
#endif
		}
#ifdef DEBUG
		file_id++;
		FeatureImage.close();
#endif

		int totalhit = 0;
		int totalsample = 0;
		int save = 0;
		double gini = 0;
		double minGini = 1;
		for (int img = 0; img < img_cnt; img++)
		{
			if (ImageFeature[img].fv > 0)
			{
				totalhit += ImageFeature[img].hit;
				totalsample++;
			}
		}
		for (int img = 0; img < img_cnt; img++)
		{
			double EN = 0;
			double EP = 0;
			if (ImageFeature[img].fv > 0)
			{
				int th = ImageFeature[img].fv;
				int errorNeg = 0;
				int errorPos = 0;
				for (int idx = 0; idx < img_cnt; idx++)
				{
					if (ImageFeature[idx].fv > 0)
					{
						if (ImageFeature[idx].fv < th &&
							ImageFeature[idx].hit == 1)
							errorPos++;
						else if (ImageFeature[idx].fv >= th &&
								 ImageFeature[idx].hit == 0)
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
	FeatureImageList.close();

	ifstream FeatureImageList1(FeatureImageFilename.c_str(), std::ifstream::binary);
	for (int fid = 0; fid < FeatureLen; fid++)
	{
		ImageFeature.clear();
		for (int img = 0; img < img_cnt; img++)
		{
			FeatureImageList1.read((char*)& Feature_tmp, sizeof(FeatureValue));
			ImageFeature.push_back(Feature_tmp);
		}
		int totalhit = 0;
		int totalsample = 0;
		int save = 0;
		double gini = 0;
		double minGini = 1;		
		for (int img = 0; img < img_cnt; img++)
		{
			if (ImageFeature[img].fv <= 0)
			{
				totalhit += ImageFeature[img].hit;
				totalsample++;
			}
		}

		for (int img = 0; img < img_cnt; img++)
		{
			double EN = 0;
			double EP = 0;
			if (ImageFeature[img].fv <= 0)
			{
				int th = ImageFeature[img].fv;
				int errorNeg = 0;
				int errorPos = 0;
				for (int idx = 0; idx < img_cnt; idx++)
				{
					if (ImageFeature[idx].fv <= 0)
					{
						if (ImageFeature[idx].fv > th &&
							ImageFeature[idx].hit == 1)
							errorPos++;
						else if (ImageFeature[idx].fv <= th &&
							ImageFeature[idx].hit == 0)
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
	FeatureList.close();
	FeatureImageList1.close();

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

void BuildThresholdHit(string FeatureImageFilename, int** ThresholdHit, vector <FeatureThreshold> ThresholdTable, int img_cnt)
{
	ifstream FeatureImageList(FeatureImageFilename.c_str(), std::ifstream::binary);
	FeatureValue Feature_tmp;
	for (int fid = 0; fid < ThresholdTable.size(); fid++)
	{
		for (int img = 0; img < img_cnt; img++)
		{
			FeatureImageList.read((char*)& Feature_tmp, sizeof(FeatureValue));
			if ((Feature_tmp.fv < ThresholdTable[fid].thp && Feature_tmp.fv > ThresholdTable[fid].thn) ||
				(Feature_tmp.fv >= 0 && ThresholdTable[fid].thp == 0) || (Feature_tmp.fv <= 0 && ThresholdTable[fid].thn == 0))
				ThresholdHit[fid][img] = 0;
			else
				ThresholdHit[fid][img] = 1;
		}
	}
}


void initWeights(vector <FeatureValue> ImageFeature, double** weights,
	             vector <FeatureThreshold> ThresholdTable, int img_cnt)
{
	for (int fid = 0; fid < ThresholdTable.size(); fid++)
	{
		int negcnt = 0;
		int poscnt = 0;
		for (int img = 0; img < img_cnt; img++)
		{
			if (ImageFeature[img * ThresholdTable.size() + fid].hit == 0) negcnt++;
			else poscnt++;
		}
		for (int img = 0; img < img_cnt; img++)
		{
			if (ImageFeature[img * ThresholdTable.size() + fid].hit == 0)
			{
				if (poscnt == 0) weights[fid][img] = 1.0 / negcnt;
				else weights[fid][img] = 1.0 / (negcnt * 2);
			}
			else
			{
				if (negcnt == 0) weights[fid][img] = 1.0 / (poscnt);
				else weights[fid][img] = 1.0 / (poscnt * 2);
			}
		}
	}
}

void normWeights(vector <FeatureValue> ImageFeature, double** weights,
	             vector <FeatureThreshold> ThresholdTable, int img_cnt)
{
	double weightSum = 0;
	for (int f = 0; f < ThresholdTable.size(); f++)
	{
		for (int i = 0; i < img_cnt; i++) weightSum += weights[f][i];
	}
	for (int f = 0; f < ThresholdTable.size(); f++)
	{
		for (int i = 0; i < img_cnt; i++) weights[f][i] /= weightSum;
	}
}

TrainOut train(std::ofstream &TableOut, double** weights, int** ThresholdHit,
	string FeatureImageFilename, vector <FeatureThreshold> ThresholdTable)
{
	ifstream FeatureImageList(FeatureImageFilename.c_str(), std::ifstream::binary);
	FeatureValue Feature_tmp;
	vector <double> featureError;
	double alpha;
	for (int fid = 0; fid < ThresholdTable.size(); fid++)
	{
		double sum = 0;
		for (int img = 0; img < img_cnt; img++)
		{
			FeatureImageList.read((char*)& Feature_tmp, sizeof(FeatureValue));
			if (ThresholdHit[fid][img] != Feature_tmp.hit)
			{
				sum += weights[fid][img];
			}
		}
		featureError.push_back(sum);
	}
	int index = std::min_element(featureError.begin(), featureError.end()) - featureError.begin();
	double minError = *std::min_element(featureError.begin(), featureError.end());

	double beta = minError / (1 - minError);
	if (beta == 0) alpha = 50;
	else alpha = log(1 / beta);

	TableOut << setw(5) << index << setw(4) << index % 4
		<< setw(6) << ThresholdTable[index].box.xs
		<< setw(6) << ThresholdTable[index].box.ys
		<< setw(4) << ThresholdTable[index].box.xe
		<< setw(4) << ThresholdTable[index].box.ye
		<< setw(12) << ThresholdTable[index].thp
		<< setw(12) << ThresholdTable[index].thn
		<< setw(12) << alpha;


/*	fprintf(TableOut, "%5d %4d %6d %6d %4d %4d %12d %12d %12f\n",
		index, index % 4, ThresholdTable[index].box.xs, ThresholdTable[index].box.ys,
		ThresholdTable[index].box.xe, ThresholdTable[index].box.ye,
		ThresholdTable[index].thp, ThresholdTable[index].thn, alpha);*/

	TrainOut tmp;
	tmp.beta = beta;
	tmp.minidx = index;
	return tmp;
}

void updateWeights(double** weights, int** ThresholdHit, string FeatureImageFilename, vector <FeatureThreshold> ThresholdTable, int img_cnt, int minIndex, double beta)
{
	ifstream FeatureImageList(FeatureImageFilename.c_str(), std::ifstream::binary);
	FeatureValue Feature_tmp;
	for (int fid = 0; fid < ThresholdTable.size() * minIndex; fid++) FeatureImageList.read((char*)& Feature_tmp, sizeof(FeatureValue));

	for (int img = 0; img < img_cnt; img++)
	{
		FeatureImageList.read((char*)& Feature_tmp, sizeof(FeatureValue));
		if (Feature_tmp.hit == ThresholdHit[minIndex][img]) weights[minIndex][img] *= beta;
	}
	FeatureImageList.close();
}

int main()
{
#ifdef DEBUG
	string ImageFilename = WorkFolder + "Image.txt";
	ofstream Image(ImageFilename.c_str());
	string NormalFilename = WorkFolder + "Normal.txt";
	ofstream Normal(NormalFilename.c_str());
	string IntegralFilename = WorkFolder + "Integral.txt";
	ofstream Integral(IntegralFilename.c_str());
#endif

	int imgsizeW, imgsizeH, offset, file_id = 0;
	string bmpsource = WorkFolder + TrainFolder + ImgPrefix + "1.bmp";

	char* header = ReadBMP256Size(bmpsource, &imgsizeW, &imgsizeH, &offset);
	char* BMP256Header = new char[offset];
	for (int i = 0; i < offset; i++) BMP256Header[i] = header[i];

	int** img = (int**)malloc(sizeof(int) * imgsizeH);
	for (int i = 0; i < imgsizeH; i++) img[i] = (int*)malloc(sizeof(int) * imgsizeW);

	int** integral = (int**)malloc(sizeof(int) * imgsizeH);
	for (int i = 0; i < imgsizeH; i++) integral[i] = (int*)malloc(sizeof(int) * imgsizeW);

	string SourceEyeTable = WorkFolder + SourceEyeTableFilename;
	vector <TableList> LeftTable, RightTable, LeftMinMax, RightMinMax;
	GetEyeList(SourceEyeTable, LeftTable, RightTable, LeftMinMax, RightMinMax);

	vector <TableList> FeatureLoc;
	BuildFeatureLoc(FeatureLoc, LeftMinMax, RightMinMax, imgsizeW);

	string FeatureListFilename = WorkFolder + "ImageFeature.bin";
	ofstream FeatureList(FeatureListFilename.c_str(), std::ofstream::binary);

	vector <FeatureValue> ImageFeature;
	for (int k = 0; k < img_cnt; k++)
	{
		file_id++;
		string bmpsource = WorkFolder + TrainFolder + ImgPrefix + to_string(file_id) + ".bmp";
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
		for (int i = 0; i < ImageFeature.size();i++) FeatureList.write((char*)& ImageFeature[i], sizeof(FeatureValue));

#ifdef DEBUG	
		string FeaturetableFilename = WorkFolder + "Image" + to_string(k) + "Feature.txt";
		ofstream FeatureTable(FeaturetableFilename.c_str());
		for (int i = 0; i < ImageFeature.size();i++)
			FeatureTable << ImageFeature[i].box.id << "	" << ImageFeature[i].box.xs << "	"
			<< ImageFeature[i].box.ys << "	" << ImageFeature[i].box.xe << "	"
			<< ImageFeature[i].box.ye << "	" << ImageFeature[i].fv << "	" << ImageFeature[i].hit << endl;
		FeatureTable.close();
#endif
	}
	FeatureList.close();
	for (int i = 0; i < imgsizeH; i++) free(img[i]); free(img);
	for (int i = 0; i < imgsizeH; i++) free(integral[i]); free(integral);

#ifdef DEBUG
	Image.close();
	Normal.close();
	Integral.close();
#endif
	string FeatureImageFilename = WorkFolder + "FeatureImage.bin";
	vector <FeatureThreshold> ThresholdTable;
	BuildFeatureThreshold(FeatureListFilename, FeatureImageFilename, FeatureLoc, ThresholdTable, img_cnt);

//#ifdef DEBUG
	string ThresholdTableFilename = WorkFolder + "ThresholdList.txt";
	ofstream ThresholdList(ThresholdTableFilename.c_str());
	for (int i = 0; i < ThresholdTable.size();i++)
	{
		ThresholdList << ThresholdTable[i].box.id << "	" << ThresholdTable[i].box.xs << "	"
			<< ThresholdTable[i].box.ys << "	" << ThresholdTable[i].box.xe << "	"
			<< ThresholdTable[i].box.ye << "	" << ThresholdTable[i].thp << "	" << ThresholdTable[i].thn << endl;
	}
	ThresholdList.close();
//#endif

	double** weights = (double**)malloc(sizeof(double) * ThresholdTable.size());
	for (int i = 0; i < ThresholdTable.size(); i++) weights[i] = (double*)malloc(sizeof(double) * img_cnt);

	int** ThresholdHit = (int**)malloc(sizeof(int) * ThresholdTable.size());
	for (int i = 0; i < ThresholdTable.size(); i++) ThresholdHit[i] = (int*)malloc(sizeof(int) * img_cnt);

	BuildThresholdHit(FeatureImageFilename, ThresholdHit, ThresholdTable, img_cnt);

//#ifdef DEBUG
	string ThresholdHitFilename = WorkFolder + "ThresholdHitList.txt";
	ofstream ThresholdHitList(ThresholdHitFilename.c_str());
	for (int i = 0; i < img_cnt; i++)
	{
		for (int col = 0; col < ThresholdTable.size(); col++)
		{
			ThresholdHitList << ThresholdHit[col][i] << "	";
		}
		ThresholdHitList << endl;
	}
//#endif

	initWeights(ImageFeature, weights, ThresholdTable, img_cnt);

	string OutputTable = WorkFolder + "TrainTable.txt";
	ofstream TableOut(OutputTable.c_str());

	TableOut << "INDEX TYPE XSTART YSTART XEND YEND THRESHOLD_P THRESHOLD_N Alpha" << endl;

	for (int i = 0; i < ThresholdTable.size(); i++)
	{
		normWeights(ImageFeature, weights, ThresholdTable, img_cnt);
		TrainOut minidx_beta = train(TableOut, weights, ThresholdHit, FeatureImageFilename, ThresholdTable);
		updateWeights(weights, ThresholdHit, FeatureImageFilename, ThresholdTable, img_cnt, minidx_beta.minidx, minidx_beta.beta);
	}

//	string bmpdest = WorkFolder + OutputFolder + ImgPrefix + to_string(file_id) + ".bmp";
//	WriteBMP256(bmpdest, imgsizeW, imgsizeH, offset, img, BMP256Header);
	for (int i = 0; i < ThresholdTable.size(); i++) free(weights[i]); free(weights);
	for (int i = 0; i < ThresholdTable.size(); i++) free(ThresholdHit[i]); free(ThresholdHit);
}

