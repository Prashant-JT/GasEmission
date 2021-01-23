#include "CSVReader.cpp"
#include <mkl_cblas.h>
#define path "Datos\\gt_2015.csv"

void printVectorDouble(double* mat, int m, int n) {
	printf("\t\t");
	for (int i = 0; i < m * n; i++)
	{
		if (i == (m * n) - 1) {
			printf("%4.2f\n", mat[i]);
		}
		else {
			if (mat[i] < 0) {
				printf("%4.1f, ", mat[i]);
			}
			else {
				printf("%4.2f, ", mat[i]);
			}
		}
	}
}

void printMatrix(double* mat, int m, int n) {
	for (int i = 0; i < m; i++)
	{
		printf("%4.1f\t", mat[i]);
		if ((i + 1) % n == 0) {
			printf("\n\t\t");
		}
	}
	printf("\n");
}

double* centerData(double * data, int m, int n) {
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
	
	return means;
}

double* clone(double* matrix, int size) {
	double* res = new double[size];
	memcpy(res, matrix, sizeof(double) * size);
	return res;
}

double* computeCov(double* XC, int m_, int n_) {
	MKL_INT m, k, n, alpha, beta, lda, ldb, ldc;
	CBLAS_LAYOUT layout;
	CBLAS_TRANSPOSE transA;
	CBLAS_TRANSPOSE transB;
	layout = CblasRowMajor;
	transA = CblasTrans;
	transB = CblasNoTrans;
	m = m_;
	k = n_;
	n = m_;
	alpha = 1;
	beta = 0;
	lda = k;
	ldb = n;
	ldc = n;

	//Operacion: 1 * DATA' * DATA - 0 * C{0}
	double* XC_Trans = clone(XC, m * n);
	double * Z = new double[m*n] {0};

	cblas_dgemm(layout, transA, transB, m, n, k, alpha, XC_Trans, lda, XC, ldb, beta, Z, ldc);

	printMatrix(Z, m, n);

	return NULL;
}

int main(int argc, char* argv[]){
	CSVReader reader(path, ',');
	double* dataRead = reader.readData();
	int m = reader.getM(); int n = reader.getN();
	double * dataCentered = centerData(dataRead, m, n);

	double* dataCov = computeCov(dataCentered, m, n);

	return 0;
}