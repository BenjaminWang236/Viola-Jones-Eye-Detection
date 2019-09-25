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
#include <bits/stdc++.h>
#include "imageScaling.h"
#include "timer.h"

using namespace std;
using namespace std::chrono;
#define N 2

string WorkFolder = "/Users/infin/OneDrive/NeuronBasic/Viola-Jones-Eye-Detection/";
string InFolder = "trainimg/";
string outFolder = "rescaled/";
string ImgPrefix = "trainimg_";

// void getCofactor(float A[N][N], float temp[N][N], int p, int q, int n)
// {
//     int i = 0, j = 0; 
  
//     // Looping for each element of the matrix 
//     for (int row = 0; row < n; row++) 
//     { 
//         for (int col = 0; col < n; col++) 
//         { 
//             //  Copying into temporary matrix only those element 
//             //  which are not in given row and column 
//             if (row != p && col != q) 
//             { 
//                 temp[i][j++] = A[row][col]; 
  
//                 // Row is filled, so increase row index and 
//                 // reset col index 
//                 if (j == n - 1) 
//                 { 
//                     j = 0; 
//                     i++; 
//                 } 
//             } 
//         } 
//     }
// }

// float determinant(float A[N][N], int n)
// {
//     float D = 0.0; // Initialize result 
  
//     //  Base case : if matrix contains single element 
//     if (n == 1) 
//         return A[0][0]; 
  
//     float temp[N][N]; // To store cofactors 
  
//     float sign = 1.0;  // To store sign multiplier 
  
//      // Iterate for each element of first row 
//     for (int f = 0; f < n; f++) 
//     { 
//         // Getting Cofactor of A[0][f] 
//         getCofactor(A, temp, 0, f, n); 
//         D += sign * A[0][f] * determinant(temp, n - 1); 
  
//         // terms are to be added with alternate sign 
//         sign = -sign; 
//     } 
  
//     return D; 
// }

// void adjoint(float A[N][N], float adj[N][N])
// {
//     if (N == 1) 
//     { 
//         adj[0][0] = 1; 
//         return; 
//     } 
  
//     // temp is used to store cofactors of A[][] 
//     float sign = 1.0, temp[N][N]; 
  
//     for (int i=0; i<N; i++) 
//     { 
//         for (int j=0; j<N; j++) 
//         { 
//             // Get cofactor of A[i][j] 
//             getCofactor(A, temp, i, j, N); 
  
//             // sign of adj[j][i] positive if sum of row 
//             // and column indexes is even. 
//             sign = ((i+j)%2==0)? 1.0: -1.0; 
  
//             // Interchanging rows and columns to get the 
//             // transpose of the cofactor matrix 
//             adj[j][i] = (sign)*(determinant(temp, N-1)); 
//         } 
//     } 
// }

// bool inverse(float A[N][N], float inverse[N][N])
// {
//     // Find determinant of A[][] 
//     float det = determinant(A, N); 
//     if (det == 0.0) 
//     { 
//         cout << "Singular matrix, can't find its inverse"; 
//         return false; 
//     } 
  
//     // Find adjoint 
//     float adj[N][N]; 
//     adjoint(A, adj); 
  
//     // Find Inverse using formula "inverse(A) = adj(A)/det(A)" 
//     for (int i=0; i<N; i++) 
//         for (int j=0; j<N; j++) 
//             inverse[i][j] = adj[i][j]/det; 
  
//     return true; 
// }

// template<class T>
// void display(T A[N][N])
// {
//     for (int i=0; i<N; i++) 
//     { 
//         for (int j=0; j<N; j++) 
//             cout << A[i][j] << " "; 
//         cout << endl; 
//     } 
// }

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

int rounder(float num)
{
    return fmod(num, 1.0) >= 0.5 ? (int) num + 1 : (int) num;
}

int* inverseMapping(int x, int y, int dx, int dy, int sx, int sy)
{
    // float inverseScaler[N][N] = {{((float)sx)/dx, 0.0}, {0.0, ((float)sy)/dy}};    
    float invX = ((float)sx)/dx, invY = ((float)sy)/dy;
    int* output = new int[N];
    output[0] = rounder(x * invX);
    output[0] = output[0] >= dx ? dx - 1 : output[0];
    output[0] = output[0] < 0 ? 0 : output[0];
    output[1] = rounder(y * invY);
    output[1] = output[1] >= dy ? dy - 1 : output[1];
    output[1] = output[1] < 0 ? 0 : output[1];
    // DEBUGGING
    cout << "(" << output[0] << ", " << output[1] << ")\t";
    return output;
}

int** imageScaler(int** src, int dx, int dy, int sx, int sy)
{
    int* sourceCoords = new int[N];
    int** dest = new int*[dx];
    for (int i = 0; i < dx; i++)    dest[i] = new int[dy];
    for (int i = 0; i < dx; i++)
    {
        for (int j = 0; j < dy; j++)
        {
            cout << "dest: (" << i << ", " << j << ") src: ";
            sourceCoords = inverseMapping(i, j, dx, dy, sx, sy);
            dest[i][j] = src[sourceCoords[0]][sourceCoords[1]];
        }
    }
    return dest;
}

int main(int argc, char** argv)
{
    auto start = high_resolution_clock::now();

    string WorkDrive = "C:";
	int dest_size_x = 10, dest_size_y = 10, img_start = 20, img_end = 29, img_cnt = img_end - img_start + 1;
	if (argc != 6)
	{
		cout << "WorkDrive NewSizeHeight NewSizeWidth Start End" << endl;
		return 0;
	}
	else
	{
        try
        {
            WorkDrive = argv[1];
            dest_size_x = atoi(argv[2]); dest_size_y = atoi(argv[3]);
	    	img_start = atoi(argv[4]); img_end = atoi(argv[5]); img_cnt = img_end - img_start + 1;
        }
        catch(const std::exception& e)
        {
            std::cerr << e.what() << endl;
        }
    }

    int imgsizeW, imgsizeH, offset, length;
	string bmpsource = WorkDrive + WorkFolder + InFolder + ImgPrefix + "0.bmp";
    string bmpoutput = WorkDrive + WorkFolder + outFolder + "rescale_0.bmp";
	vector <char> header;
	ReadBMP256Size(bmpsource, &imgsizeW, &imgsizeH, &offset, header);
    int** img = new int* [imgsizeH]; for (int i = 0; i < imgsizeH; i++) img[i] = new int[imgsizeW];
    int** out = new int* [dest_size_x]; for (int i = 0; i < dest_size_x; i++) out[i] = new int[dest_size_y];

    for (int k = 0; k < img_cnt; k++)
    {
        std::cout << k << " ";
		stringstream ss;
		ss << k;
		bmpsource = WorkDrive + WorkFolder + InFolder + ImgPrefix + ss.str() + ".bmp";
		bmpoutput = WorkDrive + WorkFolder + outFolder + "rescale_" + ss.str() + ".bmp";
        ReadBMP256(bmpsource, imgsizeW, imgsizeH, offset, img);
        out = imageScaler(img, dest_size_x, dest_size_y, imgsizeH, imgsizeW);
        WriteBMP256(bmpoutput, dest_size_x, dest_size_y, offset, out, header);
    }

    for (int i = 0; i < imgsizeH; i++) delete[] img[i]; delete[] img;
    for (int i = 0; i < imgsizeH; i++) delete[] out[i]; delete[] out;

    auto end = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(end-start);
    long conv = duration.count();
    cout << "\n--- " << format_duration(conv) << " ---" << endl;
	return 0;
}