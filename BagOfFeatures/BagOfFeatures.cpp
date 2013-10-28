#include <iostream>
#include <fstream>

#include <opencv2/nonfree/nonfree.hpp>

#include "common.hpp"
#include "BagOfFeatures.h"

cvi::BoF::BoF(int detector_type_)
	: svm(),
	  ndesc(0),
	  detector_type(detector_type_),
	  categories(),
	  indices(),
	  descriptors()
{
}

cvi::BoF::~BoF()
{
#ifndef SVM_OPENCV
	svm_free_and_destroy_model(&model);
	svm_destroy_param(&param);
	delete[] prob.y;
	delete[] prob.x;
	delete[] x_space;
#endif
}

void cvi::BoF::input(string filename) {
	/*
	 * 入力のファイルに書かれた画像から記述子を密に抽出する
	 */

	string  cate, imfile;
	int     npics;
	cv::Mat img;

	cv::SIFT sift;
	vector<cv::KeyPoint> keypoints;
	cv::Mat descriptor;

	this->ndesc    = 0;
	int cate_index = 0;
	ifstream ifs(filename, ios::in);
	if(!ifs.is_open()) {
		throw "Failed to load file";
		return;
	}

	while(!ifs.eof()) {
		if(!(ifs >> cate >> npics)) break;
		cout << "[ " << cate << " ]" << endl;
		cout << npics << " pictures" << endl;
		categories.push_back(cate);
		for(int i=0; i<npics; i++) {
			ifs >> imfile;
			cv::Mat img = cv::imread(imfile, CV_LOAD_IMAGE_COLOR);
			if(img.empty()) {
				cout << "Failed to load \"" << imfile << "\"" << endl;
				continue;
			}

			descset ds;
			ds.input(img, cate_index, detector_type);
			descriptors.push_back(ds);
			indices.push_back(cate_index);
			printf("[%04d] %d descriptors detected\n", i+1, ds.ndesc());
			ndesc += ds.ndesc();
		}
		cate_index++;
	}
	ifs.close();

	printf("[BoF] total %d descriptors detected\n", ndesc);
}

void cvi::BoF::clustering(int k) {
	/*
	 * 入力画像から得られた記述子をk個にクラスタリング
	 * このkがヒストグラムのビンの数に対応する
	 */

	// k-means法のためにcv::Matにデータをコピー
	int npics = (int)descriptors.size();
	int cnt   = 0;
	cv::Mat descs = cv::Mat::zeros(ndesc, 128, CV_32FC1);
	for(int i=0; i<npics; i++) {
		for(int j=0; j<descriptors[i].ndesc(); j++) {
			VectorXd& v = descriptors[i].get(j);
			for(int d=0; d<128; d++) {
				descs.at<float>(cnt, d) = (float)v(d);
			}
			cnt++;
		}
	}

	// k-means法の実行
	cv::Mat cc, labels;
	cv::TermCriteria crit(CV_TERMCRIT_EPS|CV_TERMCRIT_ITER, 50, 1.0e-5);
	cv::kmeans(descs, k, labels, crit, 1, cv::KMEANS_PP_CENTERS, cc); 

	// cv::Matからデータをコピー
	centers.reserve(cc.rows);
	for(int i=0; i<cc.rows; i++) {
		VectorXd v(cc.cols);
		for(int d=0; d<cc.cols; d++) {
			v(d) = cc.at<float>(i, d);
		}
		centers.push_back(v);
	}
	printf("[BoF] clustering OK. %d centers computed.\n", k);
}

void cvi::BoF::calcHist() {
	/*
	 * クラスタリングの結果をもとに各画像のヒストグラムを作る
	 */ 

	int nbins = (int)centers.size();
	int npics = (int)descriptors.size();
	
	histograms.reserve(npics);
	for(int i=0; i<npics; i++) {
		VectorXd h(nbins);
		for(int j=0; j<descriptors[i].ndesc(); j++) {
			VectorXd& v = descriptors[i].get(j);
			double mindist = INFF;
			int    minidx  = 0;
			for(int k=0; k<nbins; k++) {
				double dist = (v - centers[k]).norm2();
				if(mindist > dist) {
					mindist = dist;
					minidx  = k;
				}
			}
			h(minidx) += 1.0;
		}
		histograms.push_back(h);
	}
	printf("[BoF] histograms computed.\n");
}

#ifdef SVM_OPENCV

string cvi::BoF::predict(descset& ds) {
	// 入力の記述子集合からヒストグラムを作る
	int dim = centers.size();
	cv::Mat query = cv::Mat::zeros(1, dim, CV_32FC1);
	for(int i=0; i<ds.ndesc(); i++) {
		VectorXd& v = ds.get(i);
		double mindist = INFF;
		int    minidx  = 0;
		for(int k=0; k<dim; k++) {
			double dist = (v - centers[k]).norm2();
			if(mindist > dist) {
				mindist = dist;
				minidx  = k;
			}
		}
		query.at<float>(0, minidx) += 1.0;
	}

	// 結果の出力
	int res = (int)svm.predict(query);
	return categories[res];
}

void cvi::BoF::train(int svm_type, int kernel_type) {
	// 学習済みヒストグラムのデータをcv::Matにコピー
	cv::Mat hist = cv::Mat((int)histograms.size(), histograms[0].ndim(), CV_32FC1);
	for(int i=0; i<histograms.size(); i++) {
		VectorXd& h = histograms[i];
		for(int d=0; d<histograms[i].ndim(); d++) {
			hist.at<float>(i, d) = (float)h(d);
		}
	}

	// 記述子ベクトルをcv::Matにコピー
	cv::Mat answers = cv::Mat((int)indices.size(), 1, CV_32FC1);
	for(int i=0; i<indices.size(); i++) {
		answers.at<float>(i, 0) = (float)indices[i];
	}
	
	// SVMに学習させる
	CvSVMParams params;
	params.svm_type    = svm_type; 
	params.kernel_type = kernel_type;

	svm.train(hist, answers, cv::Mat(), cv::Mat(), params);
	printf("[BoF] svm learning finished.\n");
}

#endif

void cvi::BoF::save(string filename) const {
	ofstream ofs(filename, ios::out);
	ofs << "# category names" << endl;
	ofs << "category " << categories.size() << endl;
	for(int i=0; i<categories.size(); i++) {
		ofs << categories[i] << endl;
	}

	ofs << "# centers" << endl;
	int nbins = centers.size();
	int ndims = centers[0].ndim();
	ofs << "center " << nbins << " " << ndims << endl;
	for(int i=0; i<nbins; i++) {
		for(int d=0; d<ndims; d++) {
			ofs << centers[i](d) << (d == ndims-1 ? "\n" : " ");
		}
	}

	ofs << "# histograms" << endl;
	int npics = (int)histograms.size();
	ofs << "histogram " << npics << " " << nbins << endl;
	for(int i=0; i<npics; i++) {
		ofs << indices[i] << " ";
		for(int d=0; d<nbins; d++) {
			ofs << histograms[i](d) << (d == nbins-1 ? "\n" : " ");
		}
	}
	ofs.close();

	save_binary("hoge.dat");
}

void cvi::BoF::save_binary(string filename) const {
	ofstream ofs(filename, ios::out|ios::binary);
	
	// カテゴリ名の書き込み
	string cate  = "category";
	int    ncate = (int)categories.size();
	ofs.write(cate.c_str(), cate.length()+1);
	ofs.write((char*)&ncate, sizeof(int));
	for(int i=0; i<ncate; i++) {
		ofs.write(categories[i].c_str(), categories[i].length()+1);
	}

	// クラスタ中心の書き込み
	string cent  = "center";
	int    nbins = centers.size();
	int    ndims = centers[0].ndim();
	ofs.write(cent.c_str(), cent.length()+1);
	ofs.write((char*)&nbins, sizeof(int));
	ofs.write((char*)&ndims, sizeof(int));
	for(int i=0; i<nbins; i++) {
		for(int d=0; d<ndims; d++) {
			double val = centers[i](d);
			ofs.write((char*)&val, sizeof(double));
		}
	}

	// ヒストグラムの書き込み
	string hist  = "histogram";
	int    npics = (int)histograms.size();
	ofs.write(hist.c_str(), hist.length()+1);
	ofs.write((char*)&npics, sizeof(int));
	ofs.write((char*)&nbins, sizeof(int));
	for(int i=0; i<npics; i++) {
		ofs.write((char*)&indices[i], sizeof(int));
		for(int d=0; d<nbins; d++) {
			double val = histograms[i](d);
			ofs.write((char*)&val, sizeof(double));
		}
	}

	// 終了記号の書き込み
	string endsign = "END";
	ofs.write(endsign.c_str(), endsign.length()+1);

	ofs.close();
}

void cvi::BoF::load(string filename) {
	ifstream ifs(filename, ios::in);
	if(!ifs.is_open()) {
		throw "Failed to load file";
		return;
	}

	categories.clear();
	indices.clear();
	centers.clear();
	histograms.clear();

	string name;
	string line;
	while(getline(ifs, line)) {
		if(line[0] == '#') continue;
		stringstream ss;
		ss << line;
		ss >> name;
		if(name == "category") {
			int ncate;
			string cate;
			ss >> ncate;
			printf("[BoF] %d categories\n", ncate);
			for(int i=0; i<ncate; i++) {
				getline(ifs, line);
				ss.clear();
				ss << line;
				ss >> cate;
				categories.push_back(cate);
				cout << cate << endl;
			}
		}
		else if(name == "center") {
			int nbins, ndims;
			double val;
			ss >> nbins >> ndims;
			printf("[BoF] %d centers, %d dimensions\n", nbins, ndims);
			centers.reserve(nbins);
			for(int i=0; i<nbins; i++) {
				getline(ifs, line);
				ss.clear();
				ss << line;

				VectorXd v(ndims);
				for(int d=0; d<ndims; d++) {
					ss >> val;
					v(d) = val;
				}
				centers.push_back(v);
			}
		}
		else if(name == "histogram") {
			int id, npics, nbins;
			double val;
			ss >> npics >> nbins;
			printf("[BoF] %d histograms, %d dimensions\n", npics, nbins);
			histograms.reserve(npics);
			for(int i=0; i<npics; i++) {
				getline(ifs, line);
				ss.clear();
				ss << line;

				ss >> id;
				indices.push_back(id);
				VectorXd v(nbins);
				for(int d=0; d<nbins; d++) {
					ss >> val;
					v(d) = val;
				}
				histograms.push_back(v);
			}
		}
	}

}

void cvi::BoF::load_binary(string filename) {
	ifstream ifs(filename, ios::in|ios::binary);
	if(!ifs.is_open()) {
		throw "Failed to open file";
		return;
	}

	categories.clear();
	indices.clear();
	centers.clear();
	histograms.clear();

	string name;
	while(!ifs.eof()) {
		name = get_string(ifs);
		if(name == "END") break;

		if(name == "category") {
			int ncate;
			string cate;
			ifs.read((char*)&ncate, sizeof(int));
			for(int i=0; i<ncate; i++) {
				cate = get_string(ifs);
				categories.push_back(cate);
			}
		}
		else if(name == "center") {
			int nbins, ndims;
			double val;

			ifs.read((char*)&nbins, sizeof(int));
			ifs.read((char*)&ndims, sizeof(int));
			printf("[BoF] %d centers, %d dimensions\n", nbins, ndims);
			centers.reserve(nbins);
			for(int i=0; i<nbins; i++) {
				VectorXd v(ndims);
				for(int d=0; d<ndims; d++) {
					ifs.read((char*)&val, sizeof(double));
					v(d) = val;
				}
				centers.push_back(v);
			}
		}
		else if(name == "histogram") {
			int npics, nbins, id;
			double val;

			ifs.read((char*)&npics, sizeof(int));
			ifs.read((char*)&nbins, sizeof(int));
			printf("[BoF] %d histograms, %d dimensions\n", npics, nbins);
			histograms.reserve(npics);
			for(int i=0; i<npics; i++) {
				ifs.read((char*)&id, sizeof(int));
				indices.push_back(id);
				VectorXd v(nbins);
				for(int d=0; d<nbins; d++) {
					ifs.read((char*)&val, sizeof(double));
					v(d) = val;
				}
				histograms.push_back(v);
			}
		}
	}
	ifs.close();
}
