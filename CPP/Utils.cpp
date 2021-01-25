#include <string>
#include <mkl.h>

class Utils {
public:
	void printMatrix(double* mat, int m, int n) {
		printf("\t\t");
		for (int i = 0; i < m * n; i++)
		{
			printf("%4.8f\t", mat[i]);
			if ((i + 1) % n == 0) {
				printf("\n\t\t");
			}
		}
		printf("\n");
	}


	double* clone(double* matrix, int size) {
		double* res = new double[size];
		memcpy(res, matrix, sizeof(double) * size);
		return res;
	}

	double * centerData(double * data, int m, int n) {
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

		return data;
	}

	double * computeCov(double * data, int m_, int n_) {
		MKL_INT m, k, n, alpha, beta, lda, ldb, ldc;
		CBLAS_LAYOUT layout;
		CBLAS_TRANSPOSE transA;
		CBLAS_TRANSPOSE transB;
		layout = CblasRowMajor;
		transA = CblasTrans;
		transB = CblasNoTrans;
		m = n_; // Número de filas de C, y filas de A.	
		n = n_; // Número de columnas de C, y columnas de B.
		k = m_; // Número de columnas de A, y filas de B.
		alpha = 1; // Cálculo de la matriz de covarianza, división entre número de filas.
		beta = 0;
		lda = m;
		ldb = n;
		ldc = m;

		//Operacion: 1 * XC_Trans[11x7384]' * XC[7384x11] - 0 * Z[11x11]
		double* data_Trans = clone(data, m_ * n_);
		double * Z = new double[n_ * n_]{ 0.0 }; // Se inicializa la matriz de covarianza global.

		cblas_dgemm(layout, transA, transB, m, n, k, alpha, data_Trans, lda, data, ldb, beta, Z, ldc);

		for (int i = 0; i < n_ * n_; i++) Z[i] /= m_; // Matriz de covarianza global.

		return Z;
	}
};