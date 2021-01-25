#include <string>
#include <mkl.h>
#include <omp.h>

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

		// Calcula la media de cada columna y resta a cada componente
		// Cada columna es calculada por una seccion
		#pragma omp parallel num_threads(11)
		{
			int id = omp_get_thread_num();

			// probar clausula private(sum)
			#pragma omp sections nowait
			{
				#pragma omp section
				{
					//printf("-- [Proceso %d] -- [section 1]\n", id);
					double sum = 0;
					for (int i = 0; i < m; i++) {
						sum += data[i * n];
					}
					means[0] = sum / m;

					for (int i = 0; i < m; i++) {
						data[i * n] -= means[0];
					}
				}

				#pragma omp section
				{
					//printf("-- [Proceso %d] -- [section 2]\n", id);
					double sum = 0;
					for (int i = 0; i < m; i++) {
						sum += data[i * n + 1];
					}
					means[1] = sum / m;

					for (int i = 0; i < m; i++) {
						data[i * n + 1] -= means[1];
					}
				}

				#pragma omp section
				{
					//printf("-- [Proceso %d] -- [section 3]\n", id);
					double sum = 0;
					for (int i = 0; i < m; i++) {
						sum += data[i * n + 2];
					}
					means[2] = sum / m;

					for (int i = 0; i < m; i++) {
						data[i * n + 2] -= means[2];
					}
				}

				#pragma omp section
				{
					//printf("-- [Proceso %d] -- [section 4]\n", id);
					double sum = 0;
					for (int i = 0; i < m; i++) {
						sum += data[i * n + 3];
					}
					means[3] = sum / m;

					for (int i = 0; i < m; i++) {
						data[i * n + 3] -= means[3];
					}
				}

				#pragma omp section
				{
					//printf("-- [Proceso %d] -- [section 5]\n", id);
					double sum = 0;
					for (int i = 0; i < m; i++) {
						sum += data[i * n + 4];
					}
					means[4] = sum / m;

					for (int i = 0; i < m; i++) {
						data[i * n + 4] -= means[4];
					}
				}

				#pragma omp section
				{
					//printf("-- [Proceso %d] -- [section 6]\n", id);
					double sum = 0;
					for (int i = 0; i < m; i++) {
						sum += data[i * n + 5];
					}
					means[5] = sum / m;

					for (int i = 0; i < m; i++) {
						data[i * n + 5] -= means[5];
					}
				}

				#pragma omp section
				{
					//printf("-- [Proceso %d] -- [section 7]\n", id);
					double sum = 0;
					for (int i = 0; i < m; i++) {
						sum += data[i * n + 6];
					}
					means[6] = sum / m;

					for (int i = 0; i < m; i++) {
						data[i * n + 6] -= means[6];
					}
				}

				#pragma omp section
				{
					//printf("-- [Proceso %d] -- [section 8]\n", id);
					double sum = 0;
					for (int i = 0; i < m; i++) {
						sum += data[i * n + 7];
					}
					means[7] = sum / m;

					for (int i = 0; i < m; i++) {
						data[i * n + 7] -= means[7];
					}
				}

				#pragma omp section
				{
					//printf("-- [Proceso %d]  -- [section 9]\n", id);
					double sum = 0;
					for (int i = 0; i < m; i++) {
						sum += data[i * n + 8];
					}
					means[8] = sum / m;

					for (int i = 0; i < m; i++) {
						data[i * n + 8] -= means[8];
					}
				}

				#pragma omp section
				{
					//printf("-- [Proceso %d] -- [section 10]\n", id);
					double sum = 0;
					for (int i = 0; i < m; i++) {
						sum += data[i * n + 9];
					}
					means[9] = sum / m;

					for (int i = 0; i < m; i++) {
						data[i * n + 9] -= means[9];
					}
				}

				#pragma omp section
				{
					//printf("-- [Proceso %d]  -- [section 11]\n", id);
					double sum = 0;
					for (int i = 0; i < m; i++) {
						sum += data[i * n + 10];
					}
					means[10] = sum / m;

					for (int i = 0; i < m; i++) {
						data[i * n + 10] -= means[10];
					}
				}
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

		#pragma omp parallel num_threads(8)
		{
			// Z[11x11] Matriz de covarianza global
			//#define CHUNK 11
			#pragma omp for nowait
			for (int i = 0; i < n_ * n_; i++) {
				Z[i] /= m_;
			}
		}

		return Z;
	}
};