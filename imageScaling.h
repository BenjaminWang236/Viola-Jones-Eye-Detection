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
#include <bits/stdc++.h>
#define N 2

// /*
// Credits:
//     Matrix Cofactor, Inverse, Adjacent, Determinant code snippet found on GeeksforGeeks by Ashutosh Kumar
// */

// /*Function to get cofactor of A[p][q] in temp[][]. n is current dimension of A[][] */
// void getCofactor(float A[N][N], float temp[N][N], int p, int q, int n);

// /*
// Recursive function for finding determinant of matrix. n is current dimension of A[][].
// */
// float determinant(float A[N][N], int n);

// /*
// Function to get adjoint of A[N][N] in adj[N][N].
// */
// void adjoint(float A[N][N], float adj[N][N]);

// /*
// Function to calculate and store inverse, returns false if matrix is singular 
// */
// bool inverse(float A[N][N], float inverse[N][N]);

// /*
// Generic function to display the matrix.  We use it to display 
// both adjoin and inverse. adjoin is integer matrix and inverse
// is a float.
// */
// template<class T>
// void display(T A[N][N]); 

void ReadBMP256Size(std::string filename, int* width, int* height, int* offset, std::vector <char> &header);

int ReadBMP256(std::string filename, int width, int height, int offset, int** img);

void WriteBMP256(std::string filename, int width, int height, int offset, int** img, std::vector <char> header);

/*
Round up if decimal >= 0.5, else round down to nearest integer
*/
int rounder(float num);

/*
Use inverse mapping to get the source coordinates 
for the rescaled image (destination) pixels.
*/
int* inverseMapping(int x, int y, int dx, int dy, int sx, int sy);

/*
Scale the original image to dx/dy dimensions and return it.
Inverse Mapping with Nearest-Neighbor Interpolation
*/
int** imageScaler(int** src, int dx, int dy, int sx, int sy);

/*
Testing here
*/
int main(int argc, char** argv);

