#include "PCA.cpp"
#include "CSV_Reader.cpp"
#include <time.h>

#define path "Datos\\gt_2015.csv"


double* getPCAAutos(double * dataRead, int m, int n) {
	PCA_Autos pcaAutos(dataRead, m, n); // Constructor que prepara los datos para el cálculo del PCA (Centrado y covarianza).
	pcaAutos.autos('V', 'U'); // Se calcula autovectores y autovalores, y se reordenan de mayor a menor. Matriz de covarianza.
	return pcaAutos.getPCA(); // Se obtiene los datos mediante PCA.
}

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

	//clock_t start, stop;
	double tinicial = omp_get_wtime();
	//start = clock();
	for (int i = 0; i < 1000; i++) {
		pcaAutos = getPCAAutos(dataRead, m, n);
	}
	double tfinal = omp_get_wtime();
	double tiempo = tfinal - tinicial;
	//stop = clock();
	printf("Pos {11} = %f\n", pcaAutos[11]);
	printf("Tiempo: %4.8f segundos\n", tiempo / 1000);
		
	//double* pcaSVD = getPCASVD(dataRead, m, n);
				
	return 0;
}