#include "Utils.cpp"

class PCA_Autos {
public:
	PCA_Autos(double* matrixRead, int m, int n) {
		// Dimensiones de la matriz general [7384x11]
		m_D = m; n_D = n; 
		computeData(matrixRead);
	}

	void autos(char jobz, char uplo) { 
		// m y n son iguales porque son las dimensiones de la matriz de covarianza
		lapack_int res, layout, lda;
		layout = LAPACK_ROW_MAJOR; lda = m_C;

		double* autosVectAux = utils.clone(Z, m_C * n_C);
		autosVect = new double[m_C * n_C];

		double* autosValAux = new double[m_C];
		autosVal = new double[m_C];

		res = LAPACKE_dsyev(layout, jobz, uplo, n_C, autosVectAux, lda, autosValAux);

		int cont = 0;
		for (int i = m_C - 1; i >= 0; i--) {
			autosVal[cont] = autosValAux[i];
			cont++;
		}

		cont = 1;
		for (int j = n_C - 1; j >= 0; j--) {
			for (int i = 1; i <= m_C; i++) {
				autosVect[(n_C * (i - 1)) + (cont - 1)] = autosVectAux[(i * n_C) - cont];
			}
			cont++;
		}
	}

	double* getPCA() {
		MKL_INT m, k, n, alpha, beta, lda, ldb, ldc;
		CBLAS_LAYOUT layout;
		CBLAS_TRANSPOSE transA;
		CBLAS_TRANSPOSE transB;
		layout = CblasRowMajor;
		transA = CblasNoTrans;
		transB = CblasNoTrans;
		m = m_D; // Número de filas de C, y filas de A	
		n = n_D; // Número de columnas de C, y columnas de B
		k = n_D; // Número de columnas de A, y filas de B
		alpha = 1; // Cálculo de la matriz de covarianza, división entre número de filas
		beta = 0;
		lda = n;
		ldb = n;
		ldc = n;

		//Operacion: 1 * Data[7384x11] * AutosVectores[11x11] - 0 * dataPCA[7384x11]
		double* dataPCA = new double[m_D * n_D]{ 0.0 };

		cblas_dgemm(layout, transA, transB, m, n, k, alpha, data, lda, autosVect, ldb, beta, dataPCA, ldc);

		return dataPCA;
	}

private:
	MKL_INT m_D, n_D, m_C, n_C;
	double* data, * Z, * autosVect, * autosVal;
	Utils utils;

	void computeData(double* dataRead) {
		data = utils.centerData(dataRead, m_D, n_D);// Matriz centrada de 7384x11
		Z = utils.computeCov(data, m_D, n_D); // Matriz de covarianza de 11x11
		m_C = n_D; n_C = n_D; // Dimensiones de la matriz de covarianza [11x11]
	}
};


class PCA_SVD {
public:
	PCA_SVD(double* matrixRead, int m, int n) {
		// Dimensiones de la matriz general [7384x11]
		m_D = m; n_D = n; 
		computeData(matrixRead);
	}

	void svd(char jobu, char jobvt) {
		lapack_int res, layout, lda, ldvt, ldu;
		layout = LAPACK_ROW_MAJOR; lda = m_C;
		ldvt = lda;	ldu = lda;

		double* matrixOver = utils.clone(Z, m_C * n_C);
		double* superb = new double[n_C];

		s = new double[m_C];
		u = new double[ldu*ldu];
		double * vt = new double[ldvt*ldvt];

		res = LAPACKE_dgesvd(layout, jobu, jobvt, m_C, n_C, matrixOver, lda, s, u, ldu, vt, ldvt, superb);
	}

	double* getPCA() {
		MKL_INT m, k, n, alpha, beta, lda, ldb, ldc;
		CBLAS_LAYOUT layout;
		CBLAS_TRANSPOSE transA;
		CBLAS_TRANSPOSE transB;
		layout = CblasRowMajor;
		transA = CblasNoTrans;
		transB = CblasNoTrans;
		m = m_D; // Número de filas de C, y filas de A
		n = n_D; // Número de columnas de C, y columnas de B
		k = n_D; // Número de columnas de A, y filas de B
		alpha = 1;
		beta = 0;
		lda = n;
		ldb = n;
		ldc = n;

		// Operacion: 1 * Data[7384x11] * AutosVectores[11x11] - 0 * dataPCA[7384x11]
		double* dataPCA = new double[m_D * n_D]{ 0.0 };

		cblas_dgemm(layout, transA, transB, m, n, k, alpha, data, lda, u, ldb, beta, dataPCA, ldc);

		return dataPCA;
	}


private:
	MKL_INT m_D, n_D, m_C, n_C;
	double* data, * Z, * s, * u;
	Utils utils;

	void computeData(double* dataRead) {
		data = utils.centerData(dataRead, m_D, n_D);// Matriz centrada de 7384x11
		Z = utils.computeCov(data, m_D, n_D); // Matriz de covarianza de 11x11
		m_C = n_D; n_C = n_D; // Dimensiones de la matriz de covarianza [11x11]
	}
};