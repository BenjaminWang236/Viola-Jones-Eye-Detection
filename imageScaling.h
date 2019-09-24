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

/*
Credits:
    Matrix Cofactor, Inverse, Adjacent, Determinant code snippet found on GeeksforGeeks by Ashutosh Kumar
    Matrix Multiplication code snippet found on GeeksForGeeks by Akanksha Rai
*/

/*Function to get cofactor of A[p][q] in temp[][]. n is current dimension of A[][] */
void getCofactor(float A[N][N], float temp[N][N], int p, int q, int n);

/*
Recursive function for finding determinant of matrix. n is current dimension of A[][].
*/
float determinant(float A[N][N], int n);

/*
Function to get adjoint of A[N][N] in adj[N][N].
*/
void adjoint(float A[N][N], float adj[N][N]);

/*
Function to calculate and store inverse, returns false if matrix is singular 
*/
bool inverse(float A[N][N], float inverse[N][N]);

/*
Generic function to display the matrix.  We use it to display 
both adjoin and inverse. adjoin is integer matrix and inverse
is a float.
*/
template<class T>
void display(T A[N][N]); 

/*
Use inverse mapping to get the source coordinates 
for the rescaled image pixels using matrix multiplication
*/
int** inverseMapping(int dx, int dy, int sx, int sy);

