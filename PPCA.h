
#include "linalg.h"

using namespace alglib;

class PPCA{

public:

	PPCA(double** A, int m, int n, int k, int itr);

	void cAlg_a();
	void cAlg_b();
	

	double err();
	

	double** u();
	double* w();
	double** vt();

private:

	
	complex_2d_array data;

	
	complex_2d_array Y;
	complex_2d_array Q;
	complex_2d_array B;
	
	complex_2d_array GM;

	double* Eigval;
	double** Eigvec;

	real_1d_array W;
        real_2d_array U;
        real_2d_array VT;

	int m;
	int n;
	int k;

	int itr;

	
	
	complex_2d_array QR(complex_2d_array A);
	complex_2d_array gaussM(int r, int c, long seed);
	void normalize(complex_2d_array a);

	
	double norm_L2(complex_2d_array a);
	double norm_L2(complex_2d_array a, int ind);

	real_2d_array toR(complex_2d_array A);
	complex_2d_array csub(complex_2d_array a, complex_2d_array b);
	double** toD(real_2d_array A);
	double* toD_1d(real_1d_array A);
	void center();
	double avg(complex_2d_array a, int ind);
	void dtrans();

};
