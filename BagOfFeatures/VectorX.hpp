#ifndef _VECTOR_X_HPP_
#define _VECTOR_X_HPP_

#include <cmath>
#include <vector>
using namespace std;

template <class T = double>
class VectorX {
private:
	vector<T> data;

public:
	VectorX()
		: data()
	{
	}

	VectorX(int dim)
		: data(dim, 0)
	{
	}

	VectorX(const VectorX& v)
		: data(v.data.begin(), v.data.end())
	{
	}

	VectorX& operator=(const VectorX& v) {
		this->data = v.data;
		return *this;
	}

	VectorX operator+(VectorX v) const {
		VectorX u(data.size(), 0);
		for(int i=0; i<data.size(); i++) {
			u.data[i] = data[i] + v.data[i];
		}
		return u;
	}

	VectorX operator-(VectorX v) const {
		VectorX u((int)data.size());
		for(int i=0; i<(int)data.size(); i++) {
			u.data[i] = data[i] - v.data[i];
		}
		return u;
	}

	T& operator()(int i) {
		return data[i];
	}

	T operator()(int i) const {
		return data[i];
	}

	double norm() const {
		return sqrt(norm2());
	}

	double norm2() const {
		double ret = 0.0;
		for(int i=0; i<data.size(); i++) {
			ret += data[i] * data[i];
		}
		return ret;
	}

	int ndim() const {
		return (int)data.size();
	}
};

typedef VectorX<double> VectorXd;

#endif
