#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <stdlib.h>
#include <array>
#include <filesystem>

#include <D:\Ben Wang\OneDrive\NeuronBasic\Viola-Jones-Eye-Detection\bmp.h>

using namespace std;

/*Global variables*/
vector<char> buffer;
PBITMAPFILEHEADER file_header;
PBITMAPINFOHEADER info_header;

/*Loading the BMP images into vector*/
void fill(string &fname)
{
    std::ifstream file(fname);

    if (file)
    {
        file.seekg(0, std::ios::end);
        streampos length = file.tellg();
        file.seekg(0, std::ios::beg);

        buffer.resize(length);
        file.read(&buffer[0], length);

        file_header = (PBITMAPFILEHEADER)(&buffer[0]);
        info_header = (PBITMAPINFOHEADER)(&buffer[0] + sizeof(BITMAPFILEHEADER));
    }
}

int main()
{
    string fname = "data/database0/training_set/training1.bmp";
    fill(fname);
    cout << buffer[0] << buffer[1] << endl;
    cout << "vector size: " << buffer.size() << endl;
    cout << file_header->bfSize << endl;
    cout << file_header->bfOffBits << endl;
    cout << info_header->biWidth << " " << info_header->biHeight << endl;

    return 0;
}