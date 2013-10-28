#ifndef _COMMON_HPP_
#define _COMMON_HPP_

#include <ctime>
#include <iostream>
#include <fstream>
#include <string>

using namespace std;

#define PRT(x) cout << (#x) << " = " << (x) << endl
#define PRT2(x,y) cout << (#x) << " = " << (x) << ", " << (#y) << " = " << (y) << endl

const int    INFI = 1 << 30;
const double INFF = 1.0e12; 

/* 自作タイマークラス */
class Timer {
private:
	long t1, t2;

public:
	Timer() {}
	void start() { t1 = clock_t(); }
	void stop()  { 
		t2 = clock_t();
		printf("Time = %f\n", (t2 - t1) / 1000.0);
	}
};

/* バイナリから文字列読み込み */
inline string get_string(ifstream& ifs) {
	string ret;
	char c;
	while(ifs.get(c), c!='\0') {
		ret += c;
	}
	return ret;
}

#endif
