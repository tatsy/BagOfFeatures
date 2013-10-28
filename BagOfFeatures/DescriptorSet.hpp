#ifndef _DESCRIPTOR_HPP_
#define _DESCRIPTOR_HPP_

#include <opencv2/opencv.hpp>
#include <opencv2/nonfree/nonfree.hpp>

#include "VectorX.hpp"

namespace cvi {
	enum {
		UNKNOWN_CATEGORY = -1
	};

	enum {
		SIFT_FEATURE_DETECTOR = 0x01,
		DENSE_FEATURE_DETECTOR = 0x02
	};

	class DescriptorSet {
	private:
		int index;
		vector<VectorXd> desc;

	public:
		DescriptorSet()
			: index(UNKNOWN_CATEGORY),
			  desc()
		{
		}

		DescriptorSet(int idx) 
			: index(idx),
			  desc()
		{
		}

		DescriptorSet(const DescriptorSet& ds)
			: index(ds.index),
			  desc(ds.desc.begin(), ds.desc.end())
		{
		}

		DescriptorSet& operator=(const DescriptorSet& ds) {
			this->index = ds.index;
			this->desc  = ds.desc;
			return *this;
		}

		void input(cv::Mat& img, int idx = UNKNOWN_CATEGORY, int detector_type = SIFT_FEATURE_DETECTOR) {
			this->index = idx;
			vector<cv::KeyPoint> kp;
			cv::Mat dsc;

			if(detector_type == SIFT_FEATURE_DETECTOR) {
				cv::SIFT sift(300, 3, 0.04, 15.0, 1.6);
				sift(img, cv::Mat(), kp, dsc);
			}
			else if(detector_type == DENSE_FEATURE_DETECTOR) {
				cv::DenseFeatureDetector detector;
				cv::SiftDescriptorExtractor extractor;
				detector.detect(img, kp);
				extractor.compute(img, kp, dsc);
			}

			for(int j=0; j<dsc.rows; j++) {
				VectorXd v(dsc.cols);
				for(int d=0; d<dsc.cols; d++) {
					v(d) = dsc.at<float>(j, d);
				}
				desc.push_back(v);
			}
		}

		void add(VectorXd& v) {
			desc.push_back(v);
		}

		VectorXd& get(int i) {
			return desc[i];
		}

		int cateidx() const {
			return index;
		}

		int ndesc() const {
			return (int)desc.size();
		}
	};
}

#endif
