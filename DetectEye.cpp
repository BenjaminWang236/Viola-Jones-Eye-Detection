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
#define DEBUG_FEATURE

string WorkFolder = "D:/Ben Wang/OneDrive/NeuronBasic/Viola-Jones-Eye-Detection/";
string ImageFolder = "total/";
// string ImageFolder = "trainimg3/";
string ImgOutFolder = "detected/";

int img_cnt = 10, imgsizeW = 32, imgsizeH = 32;

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
	int alpha;
};

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

		if (bits <= 8)* offset = color * 4;
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
				unsigned char value = (unsigned char)buffer[i];
				int pixel = (int)value;
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
				buffer[i] = (char)img[row][col];
				i++;
			}
		}

		bmp256.write(buffer, width * height);
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
	negbox.ys = box.ys;                         negbox.xs = box.xs; negbox.ye = box.ys + (box.ye - box.ys) / 2; negbox.xe = box.xe;
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

void GetTrainTable(string filename, vector <FeatureList>& FeatureTable)
{
	ifstream TrainTable(filename.c_str());
	string line;
	int index;
	FeatureList Table_tmp;

	if (TrainTable.is_open())
	{
		getline(TrainTable, line);
		while (!TrainTable.eof())
		{
			TrainTable >> Table_tmp.box.id >> index >> Table_tmp.box.xs >> Table_tmp.box.ys
				>> Table_tmp.box.xe >> Table_tmp.box.ye >> Table_tmp.thp >> Table_tmp.thn >> Table_tmp.alpha;

			FeatureTable.push_back(Table_tmp);
		}
	}
	TrainTable.close();
}

TableList DetectEye(ofstream& TableOut, vector <FeatureList> FeatureTable, int** integral)
{
	int fv;
	vector <int> xsmin, ysmin, xemax, yemax;
	TableList final_box;
	for (int i = 0; i < FeatureTable.size(); i++)
	{
		if (FeatureTable[i].box.id % 4 == 0) fv = Type0(FeatureTable[i].box, integral);
		else if (FeatureTable[i].box.id % 4 == 1) fv = Type1(FeatureTable[i].box, integral);
		else if (FeatureTable[i].box.id % 4 == 2) fv = Type2(FeatureTable[i].box, integral);
		else fv = Type3(FeatureTable[i].box, integral);

		if (!((fv < FeatureTable[i].thp && fv > FeatureTable[i].thp) ||
			(fv >= 0 && FeatureTable[i].thp == 0) ||
			(fv <= 0 && FeatureTable[i].thn == 0)))
		{
			xsmin.push_back(FeatureTable[i].box.xs);
			ysmin.push_back(FeatureTable[i].box.ys);
			xemax.push_back(FeatureTable[i].box.xe);
			yemax.push_back(FeatureTable[i].box.ye);

			if (TableOut.is_open())
			{
				TableOut << FeatureTable[i].box.id << "	" << FeatureTable[i].box.id % 4
					<< "	" << FeatureTable[i].box.xs
					<< "	" << FeatureTable[i].box.ys
					<< "	" << FeatureTable[i].box.xe
					<< "	" << FeatureTable[i].box.ye
					<< "	" << FeatureTable[i].thp
					<< "	" << FeatureTable[i].thn
					<< "	" << FeatureTable[i].alpha << endl;
			}
		}
	}

	if (xsmin.size() == 0) final_box.id = 0;
	else
	{
		final_box.id = 1;
		final_box.xs = *std::min_element(xsmin.begin(), xsmin.end());
		final_box.ys = *std::min_element(ysmin.begin(), ysmin.end());
		final_box.xe = *std::max_element(xemax.begin(), xemax.end());
		final_box.ye = *std::max_element(yemax.begin(), yemax.end());
	}
	return final_box;
}

bool alphaLargeSmall(FeatureList i, FeatureList j) { return (i.alpha > j.alpha); }

void WriteTrainTable(string infilename, string outfilename)
{
	ifstream inTable(infilename.c_str());
	ofstream outTable(outfilename.c_str());
	string line;
	int index;
	FeatureList Table_tmp;
	vector <FeatureList> FeatureTable0, FeatureTable;
	vector <int> fid;
	vector <double> alpha;

	if (inTable.is_open())
	{
		outTable << "INDEX TYPE XSTART YSTART XEND YEND THRESHOLD_P THRESHOLD_N Alpha" << endl;

		while (!inTable.eof())
		{
			inTable >> Table_tmp.box.id >> index >> Table_tmp.box.xs >> Table_tmp.box.ys
				>> Table_tmp.box.xe >> Table_tmp.box.ye >> Table_tmp.thp >> Table_tmp.thn >> Table_tmp.alpha;

			if ((std::find(fid.begin(), fid.end(), Table_tmp.box.id) == fid.end()) || FeatureTable.size() == 0)
			{
				fid.push_back(Table_tmp.box.id);
				FeatureTable.push_back(Table_tmp);
			}
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

int main()
{
	int imgsizeW, imgsizeH, offset, file_id = 0;

	string bmpsource = WorkFolder + ImageFolder + "img_0.bmp";

	char* header = ReadBMP256Size(bmpsource, &imgsizeW, &imgsizeH, &offset);
	char* BMP256Header = new char[offset];
	for (int i = 0; i < offset; i++) BMP256Header[i] = header[i];


	int** img = new int* [imgsizeH];
	for (int i = 0; i < imgsizeH; i++) img[i] = new int[imgsizeW];

	int** nml = new int* [imgsizeH];
	for (int i = 0; i < imgsizeH; i++) nml[i] = new int[imgsizeW];

	int** integral = new int* [imgsizeH];
	for (int i = 0; i < imgsizeH; i++) integral[i] = new int[imgsizeW];
	
	/*
	int** img = (int**)malloc(sizeof(int) * imgsizeH);
	for (int i = 0; i < imgsizeH; i++) img[i] = (int*)malloc(sizeof(int) * imgsizeW);

	int** nml = (int**)malloc(sizeof(int) * imgsizeH);
	for (int i = 0; i < imgsizeH; i++) nml[i] = (int*)malloc(sizeof(int) * imgsizeW);

	int** integral = (int**)malloc(sizeof(int) * imgsizeH);
	for (int i = 0; i < imgsizeH; i++) integral[i] = (int*)malloc(sizeof(int) * imgsizeW);
	*/

	string TrainTablefilename = WorkFolder + "TrainTable.txt";
	vector <FeatureList> FeatureTable;
	GetTrainTable(TrainTablefilename, FeatureTable);

	string TableOutfilename = WorkFolder + "TableOut.txt";
	ofstream TableOut(TableOutfilename.c_str());


	for (int k = 0; k < img_cnt; k++)
	{
		stringstream ss;
		ss << file_id;
		string bmpsource = WorkFolder + ImageFolder + "img_" + ss.str() + ".bmp";
		string bmpoutput = WorkFolder + ImgOutFolder + "detect_" + ss.str() + ".bmp";
		file_id++;

		int avg = ReadBMP256(bmpsource, imgsizeW, imgsizeH, offset, img);
		normalize(imgsizeW, imgsizeH, img, nml, avg, 8); //keep 8 bits of floating point
		integralAll(nml, integral, imgsizeW, imgsizeH);

		TableList final_box = DetectEye(TableOut, FeatureTable, integral);

		makeBox(final_box, img);
		WriteBMP256(bmpoutput, imgsizeW, imgsizeH, offset, img, BMP256Header);
	}

	string SortOutfilename = WorkFolder + "SortOut.txt";
	WriteTrainTable(TableOutfilename, SortOutfilename);

	return 0;
}