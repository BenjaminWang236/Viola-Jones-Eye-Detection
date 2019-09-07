#include "GetEyeList.h"
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
#define DEBUG_IMGFEATURE

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

// void GetEyeList(string filename, int imgsizeW, vector <TableList> &LeftTable, vector <TableList> &RightTable, vector <TableList> &LeftMinMax, vector <TableList> &RightMinMax) {
	
// 	ifstream ifs(filename.c_str());
// 	string line;
// 	TableList Table_tmp;
// 	vector <TableList> Table;
// 	vector <int> vtmp;
// 	int tmp, num = 0, type = 0;
// 	int value[5];

// 	if (ifs.is_open())
// 	{
// 		while (getline(ifs, line))
// 		{
// 			int lineLength = line.length();
// 			for (int i = 0; i < lineLength-1; i++)
// 			{
// 				char c = line[i];
// 				if (line[i] == '[')
// 				{
// 					type = 0;
// 				}
// 				else if (line[i] > 0x2F && line[i] < 0x3A)
// 				{
// 					tmp = line[i] - 0x30;
// 					num = num * 10 + tmp;
// 				}
// 				else if (line[i] == ',')
// 				{
// 					value[type] = num;
// 					type++;
// 					num = 0;
// 				}
// 				else if (line[i] == ']')
// 				{
// 					value[type] = num;
// 					Table_tmp.id = value[0];
// 					Table_tmp.xs = value[1];
// 					Table_tmp.ys = value[2];
// 					Table_tmp.xe = value[3];
// 					Table_tmp.ye = value[4];
// #ifdef DEBUG
// 					if (Table_tmp.id == 69)
// 					{
// 						int aa = 1;
// 					}
// #endif
// 					if (Table_tmp.xs < (3 + imgsizeW / 4) && Table_tmp.xe < (6 + imgsizeW / 2)) LeftTable.push_back(Table_tmp);
// 					else RightTable.push_back(Table_tmp);

// 					num = 0;
// 					type = 0;
// 				} 
// 			}
// 		}
// 	}
// 	ifs.close();
// 	//	auto result = std::minmax_element(vtmp.begin(), vtmp.end());

// 	TableList minid_tmp, maxid_tmp;;

// 	vtmp.clear();
// 	for (int i = 0; i < LeftTable.size(); i++) vtmp.push_back(LeftTable[i].xs);
// 	Table_tmp.xs = *std::min_element(vtmp.begin(), vtmp.end());
// 	value[1] = *std::max_element(vtmp.begin(), vtmp.end());
// 	minid_tmp.xs = LeftTable[std::min_element(vtmp.begin(), vtmp.end()) - vtmp.begin()].id;
// 	maxid_tmp.xs = LeftTable[std::max_element(vtmp.begin(), vtmp.end()) - vtmp.begin()].id;
// 	vtmp.clear();
// 	for (int i = 0; i < LeftTable.size(); i++) vtmp.push_back(LeftTable[i].ys);
// 	Table_tmp.ys = *std::min_element(vtmp.begin(), vtmp.end());
// 	value[2] = *std::max_element(vtmp.begin(), vtmp.end());	
// 	minid_tmp.ys = LeftTable[std::min_element(vtmp.begin(), vtmp.end()) - vtmp.begin()].id;
// 	maxid_tmp.ys = LeftTable[std::max_element(vtmp.begin(), vtmp.end()) - vtmp.begin()].id;
// 	vtmp.clear();
// 	for (int i = 0; i < LeftTable.size(); i++) vtmp.push_back(LeftTable[i].xe);
// 	Table_tmp.xe = *std::min_element(vtmp.begin(), vtmp.end());
// 	value[3] = *std::max_element(vtmp.begin(), vtmp.end());
// 	minid_tmp.xe = LeftTable[std::min_element(vtmp.begin(), vtmp.end()) - vtmp.begin()].id;
// 	maxid_tmp.xe = LeftTable[std::max_element(vtmp.begin(), vtmp.end()) - vtmp.begin()].id;
// 	vtmp.clear();
// 	for (int i = 0; i < LeftTable.size(); i++) vtmp.push_back(LeftTable[i].ye);
// 	Table_tmp.ye = *std::min_element(vtmp.begin(), vtmp.end());
// 	value[4] = *std::max_element(vtmp.begin(), vtmp.end());
// 	minid_tmp.ye = LeftTable[std::min_element(vtmp.begin(), vtmp.end()) - vtmp.begin()].id;
// 	maxid_tmp.ye = LeftTable[std::max_element(vtmp.begin(), vtmp.end()) - vtmp.begin()].id;
// 	LeftMinMax.push_back(Table_tmp);
// 	Table_tmp.xs = value[1];
// 	Table_tmp.ys = value[2];
// 	Table_tmp.xe = value[3];
// 	Table_tmp.ye = value[4];
// 	LeftMinMax.push_back(Table_tmp);
// 	LeftMinMax.push_back(minid_tmp);
// 	LeftMinMax.push_back(maxid_tmp);

// 	vtmp.clear();
// 	for (int i = 0; i < RightTable.size(); i++) vtmp.push_back(RightTable[i].xs);
// 	Table_tmp.xs = *std::min_element(vtmp.begin(), vtmp.end());
// 	value[1] = *std::max_element(vtmp.begin(), vtmp.end());
// 	minid_tmp.xs = RightTable[std::min_element(vtmp.begin(), vtmp.end()) - vtmp.begin()].id;
// 	maxid_tmp.xs = RightTable[std::max_element(vtmp.begin(), vtmp.end()) - vtmp.begin()].id;
// 	vtmp.clear();
// 	for (int i = 0; i < RightTable.size(); i++) vtmp.push_back(RightTable[i].ys);
// 	Table_tmp.ys = *std::min_element(vtmp.begin(), vtmp.end());
// 	value[2] = *std::max_element(vtmp.begin(), vtmp.end());
// 	minid_tmp.ys = RightTable[std::min_element(vtmp.begin(), vtmp.end()) - vtmp.begin()].id;
// 	maxid_tmp.ys = RightTable[std::max_element(vtmp.begin(), vtmp.end()) - vtmp.begin()].id;
// 	vtmp.clear();
// 	for (int i = 0; i < RightTable.size(); i++) vtmp.push_back(RightTable[i].xe);
// 	Table_tmp.xe = *std::min_element(vtmp.begin(), vtmp.end());
// 	value[3] = *std::max_element(vtmp.begin(), vtmp.end());
// 	minid_tmp.xe = RightTable[std::min_element(vtmp.begin(), vtmp.end()) - vtmp.begin()].id;
// 	maxid_tmp.xe = RightTable[std::max_element(vtmp.begin(), vtmp.end()) - vtmp.begin()].id;
// 	vtmp.clear();
// 	for (int i = 0; i < RightTable.size(); i++) vtmp.push_back(RightTable[i].ye);
// 	Table_tmp.ye = *std::min_element(vtmp.begin(), vtmp.end());
// 	value[4] = *std::max_element(vtmp.begin(), vtmp.end());
// 	minid_tmp.ye = RightTable[std::min_element(vtmp.begin(), vtmp.end()) - vtmp.begin()].id;
// 	maxid_tmp.ye = RightTable[std::max_element(vtmp.begin(), vtmp.end()) - vtmp.begin()].id;
// 	RightMinMax.push_back(Table_tmp);
// 	Table_tmp.xs = value[1];
// 	Table_tmp.ys = value[2];
// 	Table_tmp.xe = value[3];
// 	Table_tmp.ye = value[4];
// 	RightMinMax.push_back(Table_tmp);
// 	RightMinMax.push_back(minid_tmp);
// 	RightMinMax.push_back(maxid_tmp);

// #ifdef DEBUG
// 	string EyeListFilename = WorkFolder + "EyeList.txt";
// 	ofstream EyeList(EyeListFilename.c_str());
// 	EyeList << "[";
// 	int Li = 0, Ri = 0;
// 	while(Li < LeftTable.size() && Ri < RightTable.size())
// 	{
// 		if (LeftTable[Li].id <= RightTable[Ri].id)
// 		{
// 			EyeList << "[" << (LeftTable[Li].id) << ", "
// 				<< (LeftTable[Li].xs) << ", " << (LeftTable[Li].ys) << ", "
// 				<< (LeftTable[Li].xe) << ", " << (LeftTable[Li].ye) << "], ";
// 			Li++;
// 		}
// 		else
// 		{
// 			EyeList << "[" << (RightTable[Ri].id) << ", "
// 				<< (RightTable[Ri].xs) << ", " << (RightTable[Ri].ys) << ", "
// 				<< (RightTable[Ri].xe) << ", " << (RightTable[Ri].ye) << "], ";
// 			Ri++;
// 		}
// 	}
// 	for (int i = Li; i < LeftTable.size();i++)
// 	{
// 		EyeList << "[" << (LeftTable[i].id) << ", "
// 			<< (LeftTable[i].xs) << ", " << (LeftTable[i].ys) << ", "
// 			<< (LeftTable[i].xe) << ", " << (LeftTable[i].ye) << "]";
// 	}
// 	for (int i = Ri; i < RightTable.size();i++)
// 	{
// 		EyeList << "[" << (RightTable[i].id) << ", "
// 			<< (RightTable[i].xs) << ", " << (RightTable[i].ys) << ", "
// 			<< (RightTable[i].xe) << ", " << (RightTable[i].ye) << "]";
// 	}
// 	EyeList << "]";
// 	EyeList.close();
// #endif
// 	cout << "[Left Eye]" << endl;
// 	cout << "Image " << LeftMinMax[2].xs << " : xs_min = " << LeftMinMax[0].xs 
// 		<< ", Image " << LeftMinMax[2].ys << " : ys_min = " << LeftMinMax[0].ys
// 		<< ", Image " << LeftMinMax[2].xe << " : xe_min = " << LeftMinMax[0].xe
// 		<< ", Image " << LeftMinMax[2].ye << " : ye_min = " << LeftMinMax[0].ye << endl;

// 	cout << "Image " << LeftMinMax[3].xs << " : xs_max = " << LeftMinMax[1].xs
// 		<< ", Image " << LeftMinMax[3].ys << " : ys_max = " << LeftMinMax[1].ys
// 		<< ", Image " << LeftMinMax[3].xe << " : xe_max = " << LeftMinMax[1].xe
// 		<< ", Image " << LeftMinMax[3].ye << " : ye_max = " << LeftMinMax[1].ye << endl;

// 	cout << "[Right Eye]" << endl;
// 	cout << "Image " << RightMinMax[2].xs << " : xs_min = " << RightMinMax[0].xs
// 		<< ", Image " << RightMinMax[2].ys << " : ys_min = " << RightMinMax[0].ys
// 		<< ", Image " << RightMinMax[2].xe << " : xe_min = " << RightMinMax[0].xe
// 		<< ", Image " << RightMinMax[2].ye << " : ye_min = " << RightMinMax[0].ye << endl;

// 	cout << "Image " << RightMinMax[3].xs << " : xs_max = " << RightMinMax[1].xs
// 		<< ", Image " << RightMinMax[3].ys << " : ys_max = " << RightMinMax[1].ys
// 		<< ", Image " << RightMinMax[3].xe << " : xe_max = " << RightMinMax[1].xe
// 		<< ", Image " << RightMinMax[3].ye << " : ye_max = " << RightMinMax[1].ye << endl;
// }