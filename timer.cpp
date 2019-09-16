#include "timer.h"
#include <chrono>
#include <sstream>
#include <string>

using namespace std;
using namespace std::chrono;

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