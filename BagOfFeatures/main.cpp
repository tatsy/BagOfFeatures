#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>
#include <map>
using namespace std;

#include <opencv2/opencv.hpp>
#include <opencv2/nonfree/nonfree.hpp>

#include "BagOfFeatures.h"

const string train_file = "./Data/train.txt";
const string test_file  = "./Data/test.txt";
const string hist_file  = "./hist_sparse.dat";

const int detector_type = cvi::SIFT_FEATURE_DETECTOR;

int main(int argc, char** argv) {
	cv::initModule_nonfree();

	cvi::BoF bof(detector_type);
	string choice;
	cout << "load pre-computed data? [Y or n]:";
	cin >> choice;
	if(choice == "n") {
		bof.input(train_file);
		bof.clustering(150);
		bof.calcHist();
		bof.save_binary(hist_file);
	} else {
		bof.load_binary(hist_file);
	}

	bof.train(cv::SVM::C_SVC, cv::SVM::LINEAR);

	int n_correct = 0;
	int n_total   = 0;

	int npic;
	string cate, imfile;
	ifstream ifs(test_file, ios::in);
	while(ifs >> cate >> npic) {
		for(int i=0; i<npic; i++) {
			ifs >> imfile;
			cv::Mat query = cv::imread(imfile, CV_LOAD_IMAGE_COLOR);
			if(query.empty()) {
				cout << "Failed to load file \"" << imfile << "\"" << endl;
				continue;
			}

			descset ds;
			ds.input(query, -1, detector_type);
			string pre = bof.predict(ds);
			printf("[BoF] predict [%s], answer [%s], %s\n", pre.c_str(), cate.c_str(), (pre == cate ? "OK" : "NG")); 
			
			if(pre == cate) n_correct++;
			n_total++;
		}
	}
	printf("[BoF] %d trials, %d correct prediction, correct rate is %f.\n", n_total, n_correct, (double)n_correct / (double)n_total);
}
