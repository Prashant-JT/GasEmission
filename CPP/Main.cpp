#include "PCA.cpp"
#include "CSV_Reader.cpp"

#include <time.h>

#define path "Datos\\gt_2015.csv"

/*
	JOBZ is CHARACTER*1
		  = 'N':  Compute eigenvalues only;
		  = 'V':  Compute eigenvalues and eigenvectors.
	UPLO is CHARACTER*1
		  = 'U':  Upper triangle of A is stored;
		  = 'L':  Lower triangle of A is stored.
*/
double* getPCAAutos(double * dataRead, int m, int n) {
	PCA_Autos pcaAutos(dataRead, m, n); // Constructor que prepara los datos para el cálculo del PCA (Centrado y covarianza).
	pcaAutos.autos('V', 'U'); // Se calcula autovectores y autovalores, y se reordenan de mayor a menor. Matriz de covarianza.
	return pcaAutos.getPCA(); // Se obtiene los datos mediante PCA.
}

/*
	JOBU is CHARACTER*1
		  Specifies options for computing all or part of the matrix U:
		  = 'A':  all M columns of U are returned in array U:
		  = 'S':  the first min(m,n) columns of U (the left singular
				  vectors) are returned in the array U;
		  = 'O':  the first min(m,n) columns of U (the left singular
				  vectors) are overwritten on the array A;
		  = 'N':  no columns of U (no left singular vectors) are
				  computed.

	JOBVT is CHARACTER*1
		  Specifies options for computing all or part of the matrix
		  V**T:
		  = 'A':  all N rows of V**T are returned in the array VT;
		  = 'S':  the first min(m,n) rows of V**T (the right singular
				  vectors) are returned in the array VT;
		  = 'O':  the first min(m,n) rows of V**T (the right singular
				  vectors) are overwritten on the array A;
		  = 'N':  no rows of V**T (no right singular vectors) are
				  computed.
		  JOBVT and JOBU cannot both be 'O'.
		  Defecto ambos igual a 'A'.
	*/

double* getPCASVD(double* dataRead, int m, int n) {
	PCA_SVD pcaSVD(dataRead, m, n); // Constructor que prepara los datos para el cálculo del PCA (Centrado y covarianza).
	pcaSVD.svd('A', 'A'); // Se calcula sus valores y vectores singulares.
	return pcaSVD.getPCA(); // Se obtiene los datos mediante PCA.
}

int main(int argc, char* argv[]){
	CSV_Reader reader(path, ',');
	Utils utils;
	double* dataRead = reader.readData();	
	int m = reader.getM(); int n = reader.getN();

	double* pcaAutos;

	clock_t start, stop;
	start = clock();
	for (int i = 0; i < 1000; i++) {
		pcaAutos = getPCAAutos(dataRead, m, n);
	}
	stop = clock();
	printf("Pos {11} = %f\n", pcaAutos[11]);
	printf("Tiempo: %4.8f segundos\n", (double)(stop - start) / CLOCKS_PER_SEC / 1000);
		
	//double* pcaSVD = getPCASVD(dataRead, m, n);
				
	return 0;
}