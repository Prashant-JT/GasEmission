#include "CSVReader.cpp"
#include "PCA.cpp"
#include <cstdio>
#include <cstdlib>

#define path "Datos\\gt_2015.csv"

/*
	JOBZ is CHARACTER*1
		  = 'N':  Compute eigenvalues only;
		  = 'V':  Compute eigenvalues and eigenvectors.
	UPLO is CHARACTER*1
		  = 'U':  Upper triangle of A is stored;
		  = 'L':  Lower triangle of A is stored.
	Se pide ambos, por tanto se pasa jobz='V', uplo=L o U (traspuestas).
	Como jobz='V', la matriz contiene los autovectores.
	Los autovalores se sitúan en el vector "w".
	*/

int main(int argc, char* argv[]){
	CSVReader reader(path, ',');	
	double* dataRead = reader.readData();	
	int m = reader.getM(); int n = reader.getN();
	PCA pca(dataRead);
	pca.centerData(m, n);	// Matriz de 7384x11
	pca.computeCov(m, n); // Matriz de 7384x11
	pca.autos('V', 'U', n, n); // Se calcula autovectores y autovalores, y se reordenan de mayor a menor
	double * dataPCA = pca.getPCA(m, n); // Se obtiene los datos mediante PCA.
		
	return 0;
}