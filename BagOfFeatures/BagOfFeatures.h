#ifndef _BAG_OF_FEATURES_H_
#define _BAG_OF_FEATURES_H_

#include <vector>
#include <string>
using namespace std;

#include <opencv2/opencv.hpp>
#include "svm.h"

#include "DescriptorSet.hpp"

typedef cvi::DescriptorSet descset;
struct svm_model;

#define SVM_OPENCV 1

namespace cvi {
	class BoF {
	private:
		int ndesc;
		int detector_type;

#ifdef SVM_OPENCV
		cv::SVM svm;
#else		
		struct svm_parameter param;
		struct svm_problem   prob;
		struct svm_model *model;
		struct svm_node  *x_space;
#endif

	public:
		vector<string>   categories;
		vector<int>      indices;
		vector<descset>  descriptors;
		vector<VectorXd> centers;
		vector<VectorXd> histograms;

	public:
		BoF(int detector_type = cvi::SIFT_FEATURE_DETECTOR);
		virtual ~BoF();

		void input(string filename);
		void clustering(int k);
		void calcHist();

		void save(string filename) const;
		void load(string filename);
		void save_binary(string filename) const;
		void load_binary(string filename);

		void train(int svm_type = cv::SVM::C_SVC, int kernel_type = cv::SVM::LINEAR);
		string predict(descset& ds);		
	};
}

#endif
