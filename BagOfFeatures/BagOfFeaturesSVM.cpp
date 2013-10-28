#include "common.hpp"
#include "BagOfFeatures.h"

#ifndef SVM_OPENCV

void cvi::BoF::train() {
	const int NUM_OF_DATA_SET = (int)indices.size();
	const int MAX_INDEX       = (int)centers.size();

	param.svm_type     = C_SVC;
	param.kernel_type  = RBF;
	param.degree       = 5;
	param.gamma        = 1.0 / MAX_INDEX;
	param.coef0        = 0;
	param.nu           = 0.5;
	param.cache_size   = 100;
	param.C            = 1;
	param.eps          = 1.0e-5;
	param.p            = 0.1;
	param.shrinking    = 1;
	param.probability  = 0;
	param.nr_weight    = 0;
	param.weight_label = NULL;
	param.weight       = NULL;

	prob.l = NUM_OF_DATA_SET;
	prob.y = new double[prob.l];
	prob.x = new svm_node*[prob.l];
	x_space = new svm_node[(MAX_INDEX+1)*prob.l];

	for(int i=0; i<prob.l; i++) {
		prob.y[i] = indices[i];
		for(int j=0; j<MAX_INDEX; j++) {
			x_space[(MAX_INDEX+1)*i + j].index = j + 1;
			x_space[(MAX_INDEX+1)*i + j].value = histograms[i](j);
		}
		x_space[(MAX_INDEX+1)*i + MAX_INDEX].index = -1;
		prob.x[i] = &x_space[(MAX_INDEX+1)*i];
	}

	model = svm_train(&prob, &param);
	svm_save_model("model.txt", model);
	printf("[BoF] svm learning finished.\n");
}

string cvi::BoF::predict(descset& ds) {
	const int MAX_INDEX       = centers.size();

	// 入力の記述子集合からヒストグラムを作る
	struct svm_node* query;
	query = new svm_node[MAX_INDEX+1];

	for(int i=0; i<ds.ndesc(); i++) {
		VectorXd& v = ds.get(i);
		double mindist = INFF;
		int    minidx  = 0;
		for(int k=0; k<MAX_INDEX; k++) {
			double dist = (v - centers[k]).norm2();
			if(mindist > dist) {
				mindist = dist;
				minidx  = k;
			}
		}
		query[minidx].value += 1.0;
	}

	for(int j=0; j<MAX_INDEX; j++) {
		query[j].index = j + 1;
	}
	query[MAX_INDEX].index = -1;

	// 結果の出力
	int res = (int)svm_predict(model, query);
	delete[] query;

	return categories[res];
}

#endif
