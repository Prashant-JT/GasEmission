#include "Utils.cpp"

class PCA_SVD {
public:
	PCA_SVD(double* matrixRead, int m, int n) {
		m_D = m; n_D = n; // Dimensiones de la matriz general [7384x11].
		computeData(matrixRead);
	}

	void svd(char jobu, char jobvt) {
		lapack_int res, layout, lda, ldvt, ldu;
		layout = LAPACK_ROW_MAJOR; lda = m_C;
		ldvt = lda;	ldu = lda;

		double* matrixOver = utils.clone(Z, m_C * n_C);		
		double* superb = new double[n_C];
		
		s = new double[m_C];
		u = new double[ldu];
		vt = new double[ldvt];

		res = LAPACKE_dgesvd(layout, jobu, jobvt, m_C, n_C, matrixOver, lda, s, u, ldu, vt, ldvt, superb);
		utils.printMatrix(matrixOver, m_C, n_C);
		printf("\t\tVector U:\n");
		utils.printMatrix(u, 1, ldu);
		printf("\t\tVector S:\n");
		utils.printMatrix(s, 1, m_C);
		printf("\t\tVector VT:\n");
		utils.printMatrix(vt, 1, ldvt);
	}


private:
	MKL_INT m_D, n_D, m_C, n_C;
	double* data, * Z, * s, * u, * vt;
	Utils utils;

	void computeData(double* dataRead) {
		data = utils.centerData(dataRead, m_D, n_D);// Matriz centrada de 7384x11.
		Z = utils.computeCov(data, m_D, n_D); // Matriz de covarianza de 11x11->Z
		m_C = n_D; n_C = n_D; // Dimensiones de la matriz de covarianza [11x11].
	}
};