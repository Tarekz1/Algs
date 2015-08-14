
#include "PPCA.h"
#include "linalg.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>


PPCA::PPCA(double** A, int r, int c, int k, int itr){

	this->data.setlength(r, c);

	for(int i=0; i<r; i++){
		for(int j=0; j<c; j++){
			data[i][j].x = A[i][j];
		    data[i][j].y = 0; }
		
	}
	////////////////
	center();
	////////////////
	//normalize(data);

	this->m=r;
	this->n=c;
	this->k=k;
	this->itr=itr;

	////////////////
	dtrans();
	////////////////
}


complex_2d_array PPCA::gaussM(int r, int c){

	alglib::hqrndstate rs;
	alglib::hqrndrandomize(rs);
	
	
	alglib::complex_2d_array gm;
	gm.setlength(r, c);
	
	for(int i=0; i<r; i++){
		for(int j=0; j<c; j++){
			gm[i][j].x = hqrndnormal(rs);
		    gm[i][j].y = 0; 
		    }
	}

	return gm;

}
void PPCA::rfe(){
cAlg_a();
cAlg_b();
}
void PPCA::cAlg_a(){

	//Y = A*GM
	this->GM = this->gaussM(n, k);
	this->Y.setlength(m, k);

	alglib::cmatrixgemm(data.rows(), GM.cols(), data.cols(), 1.0, data, 0, 0, 0, GM, 0, 0, 0, 0, Y, 0, 0);

	
	alglib::complex_2d_array tmp;
	tmp.setlength(n,k);
	alglib::complex_2d_array data_t;
	data_t.setlength(n,m);
	alglib::cmatrixtranspose(m,n,data,0,0,data_t,0,0);

	
	// alg 4.3
	for(int i=0; i<itr; i++){
	    
  alglib::cmatrixgemm(data_t.rows(), Y.cols(), data_t.cols(), 1.0, data_t, 0, 0, 0, Y, 0, 0, 0, 0, tmp, 0, 0);
	alglib::cmatrixgemm(data.rows(), tmp.cols(), data.cols(), 1.0, data, 0, 0, 0, tmp, 0, 0, 0, 0, Y, 0, 0);
	
	 }

	//QR dc
	alglib::complex_1d_array tau;
	alglib::cmatrixqr(Y, Y.rows(), Y.cols(), tau);
	alglib::cmatrixqrunpackq(Y, Y.rows(), Y.cols(), tau, Y.cols(), Q);
	// alg 4.3
	

	/*
	// alg 4.4
    Q = QR(Y);
	for(int i=1; i<itr; i++){
	alglib::cmatrixgemm(data_t.rows(), Q.cols(), data_t.cols(), 1.0, data_t, 0, 0, 0, Q, 0, 0, 0, 0, Y, 0, 0);
	Q = QR(Y);
	alglib::cmatrixgemm(data.rows(), Q.cols(), data.cols(), 1.0, data, 0, 0, 0, Q, 0, 0, 0, 0, Y, 0, 0);
	Q = QR(Y);
	}

	// alg 4.4
	*/

}

void PPCA::cAlg_b(){

	alglib::complex_2d_array Q_t;
	Q_t.setlength(Q.cols(), Q.rows());
	alglib::cmatrixtranspose(Q.rows(), Q.cols(), Q, 0, 0, Q_t, 0, 0);

	B.setlength(Q_t.rows(), data.cols());


	alglib::cmatrixgemm(Q_t.rows(), data.cols(), Q_t.cols(), 1.0, Q_t, 0, 0, 0, data, 0, 0, 0, 0, B, 0, 0);

	alglib::real_2d_array B2;
	B2.setlength(B.rows(), B.cols());
	B2=toR(B);

	//SVD
	real_2d_array U2;
	real_2d_array Q2 = toR(Q);
	alglib::rmatrixsvd(B2, B2.rows(), B2.cols(), 1, 1, 2, W, U2, VT);
	
	//U.setlength(Q2.rows(), U2.cols());
	
	//alglib::rmatrixgemm(Q2.rows(), U2.cols(), Q2.cols(), 1.0, Q2, 0, 0, 0, U2, 0, 0, 0, 0, U, 0, 0);
	//SVD
}


double** PPCA::u(){
	return toD(U);
}
double* PPCA::w(){
	return toD_1d(W);
}
double** PPCA::vt(){
	return toD(VT);
}


double PPCA::err(){
//	||(I-QQ*)A||

	alglib::complex_2d_array tmp;
	alglib::complex_2d_array tmp2;
	alglib::complex_2d_array Q_t;
	
	Q_t.setlength(Q.cols(), Q.rows());
	alglib::cmatrixtranspose(Q.rows(), Q.cols(), Q, 0, 0, Q_t, 0, 0);

	tmp.setlength(Q_t.rows(), data.cols());
	tmp2.setlength(Q.rows(), tmp.cols());
	alglib::cmatrixgemm(Q_t.rows(), data.cols(), Q_t.cols(), 1.0, Q_t, 0, 0, 0, data, 0, 0, 0, 0, tmp, 0, 0);
  alglib::cmatrixgemm(Q.rows(), tmp.cols(), Q.cols(), 1.0, Q, 0, 0, 0, tmp, 0, 0, 0, 0, tmp2, 0, 0);

  return norm_L2(csub(data,tmp2));

}

double PPCA::norm_L2(complex_2d_array a){
	double n=0.0;

	for(int i=0; i<a.rows(); i++)
		for(int j=0; j<a.cols(); j++)
			n+= pow(a[i][j].x, 2);
		
	return sqrt(n);
}

double PPCA::norm_L2(complex_2d_array a, int ind){
	double n=0.0;

	for(int i=0; i<a.rows(); i++)
			n+= pow(a[i][ind].x, 2);
		
	return sqrt(n);
}

complex_2d_array PPCA::QR(complex_2d_array A){
	alglib::complex_1d_array tau;
	alglib::complex_2d_array q;
	alglib::smp_cmatrixqr(A, A.rows(), A.cols(), tau);
	alglib::smp_cmatrixqrunpackq(A, A.rows(), A.cols(), tau, A.cols(), q);
	return q;
}




real_2d_array PPCA::toR(complex_2d_array A){

	real_2d_array A2;
	A2.setlength(A.rows(), A.cols());

	for(int i=0; i<A.rows(); i++){
		for(int j=0; j<A.cols(); j++){
			A2(i,j) = A(i,j).x;}
		
	}

	return A2;

}

complex_2d_array PPCA::csub(complex_2d_array a, complex_2d_array b){
	complex_2d_array tmp;
	tmp.setlength(a.rows(), a.cols());

	for(int i=0; i<a.rows(); i++){
		for(int j=0; j<a.cols(); j++){
			tmp[i][j].x = a[i][j].x - b[i][j].x;
		     tmp[i][j].y=0;}
		
	}

	return tmp;

}

double** PPCA::toD(real_2d_array A){

	double** tmp = (double**) malloc(A.rows()*sizeof(double));
	for(int i=0; i<A.rows(); i++)
		tmp[i] = new double[A.cols()];
	
	for(int i=0; i<A.rows(); i++){
		for(int j=0; j<A.cols(); j++){
			tmp[i][j] = A(i,j);}
	}

	return tmp;
}

double* PPCA::toD_1d(real_1d_array A){
	
	double* tmp = new double[A.length()];

	for(int i=0; i<A.length(); i++)
			tmp[i] = A(i);
	

	return tmp;
}


void PPCA::center(){

	for(int i=0; i<data.rows(); i++){
	double av = avg(data, i);
	for(int j=0; j<data.cols(); j++){
	data(i,j).x -= av;
	}
	}

}

double PPCA::avg(complex_2d_array a, int ind){
	double avg = 0.0;
	for(int i=0; i<a.cols(); i++)
	avg+= a(ind,i).x;
	
	return avg/a.cols();
}

void PPCA::normalize(complex_2d_array a){

	for(int i=0; i<a.cols(); i++){
	double nl2 = norm_L2(a, i);
	for(int j=0; j<a.rows(); j++){
	a(j,i).x /= nl2;
	}
	}
	
}


void PPCA::dtrans(){

	alglib::complex_2d_array data_t;
	data_t.setlength(n,m);
	alglib::cmatrixtranspose(m,n,data,0,0,data_t,0,0);
	data = data_t;
	int tmp=m;
	m=n;
	n=tmp;
}
