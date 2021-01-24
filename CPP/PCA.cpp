#include <cstdio>
#include <cstdlib>
#include "Utils.cpp"

#include <mkl.h>

using namespace std;

class PCA {
public:
	PCA(double * matrixRead) {
		data = matrixRead;
	}

	void centerData(int m, int n) {
		double* means = new double[n];

		double sum;
		for (int j = 0; j < n; j++) {
			sum = 0;
			for (int i = 0; i < m; i++) {
				sum += data[i * n + j];
			}

			means[j] = sum / m;

			for (int i = 0; i < m; i++) {
				data[i * n + j] -= means[j];
			}
		}
	}

	void computeCov(int m_, int n_) {
		MKL_INT m, k, n, alpha, beta, lda, ldb, ldc;
		CBLAS_LAYOUT layout;
		CBLAS_TRANSPOSE transA;
		CBLAS_TRANSPOSE transB;
		layout = CblasRowMajor;
		transA = CblasTrans;
		transB = CblasNoTrans;
		m = n_; // N�mero de filas de C, y filas de A.	
		n = n_; // N�mero de columnas de C, y columnas de B.
		k = m_; // N�mero de columnas de A, y filas de B.
		alpha = 1; // C�lculo de la matriz de covarianza, divisi�n entre n�mero de filas.
		beta = 0;
		lda = m;
		ldb = n;
		ldc = m;
		
		//Operacion: 1 * XC_Trans[11x7384]' * XC[7384x11] - 0 * Z[11x11]
		double* data_Trans = utils.clone(data, m_ * n_);
		Z = new double[n_ * n_]{ 0.0 }; // Se inicializa la matriz de covarianza global.

		cblas_dgemm(layout, transA, transB, m, n, k, alpha, data_Trans, lda, data, ldb, beta, Z, ldc);

		for (int i = 0; i < n_ * n_; i++) Z[i] /= m_; // Matriz de covarianza global.
	}
	
	void autos(char jobz, char uplo, int m, int n) { // m y n son iguales, matriz de covarianza (cuadrada).
		lapack_int res, layout, lda;
		layout = LAPACK_ROW_MAJOR; lda = m;

		double * autosVectAux = utils.clone(Z, m * n);
		autosVect = new double[m * n];

		double * autosValAux = new double[m];	
		autosVal = new double[m];

		res = LAPACKE_dsyev(layout, jobz, uplo, n, autosVectAux, lda, autosValAux);

		int cont = 0;
		for (int i = m - 1; i >= 0; i--) {
			autosVal[cont] = autosValAux[i];
			cont++;
		}

		cont = 1;
		for (int j = n-1; j >= 0; j--) {
			for (int i = 1; i <= m; i++) {
				autosVect[(n*(i-1))+(cont-1)] = autosVectAux[(i * n)-cont];
			}
			cont++;			
		}
	}

	double * getPCA(int m_, int n_) {
		MKL_INT m, k, n, alpha, beta, lda, ldb, ldc;
		CBLAS_LAYOUT layout;
		CBLAS_TRANSPOSE transA;
		CBLAS_TRANSPOSE transB;
		layout = CblasRowMajor;
		transA = CblasNoTrans;
		transB = CblasNoTrans;
		m = m_; // N�mero de filas de C, y filas de A.	
		n = n_; // N�mero de columnas de C, y columnas de B.
		k = n_; // N�mero de columnas de A, y filas de B.
		alpha = 1; // C�lculo de la matriz de covarianza, divisi�n entre n�mero de filas.
		beta = 0;
		lda = n;
		ldb = n;
		ldc = n;

		//Operacion: 1 * Data[7384x11] * AutosVectores[11x11] - 0 * dataPCA[7384x11]
		double * dataPCA = new double[m_ * n_]{ 0.0 }; 

		cblas_dgemm(layout, transA, transB, m, n, k, alpha, data, lda, autosVect, ldb, beta, dataPCA, ldc);
		
		return dataPCA;
	}

private:
	double* data, * Z, *autosVect, *autosVal;
	Utils utils;
};