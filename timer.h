#pragma once
#include <chrono>
#include <sstream>
#include <string>

/*
Takes std::chrono::duration and return it as readable hour::minutes::seconds::milliseconds::microseconds format

USAGE - 
    #include <chrono>
    using namespace std::chrono;

    //@ the top of where you want to time:
    auto start = high_resolution_clock::now();

    //@ the end of what you want to time:
    auto end = high_resolution_clock::now();
	auto duration = duration_cast<microseconds>(end-start);
	cout << "--- Execution time: "<< format_duration((long) duration.count()) << " ---" << endl;
*/
std::string format_duration(long);
