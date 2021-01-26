/*
@authors
Jose María Amusquívar Poppe
Prashant Jeswani Tejwani
*/
#include "PCA.cpp"
#include "CSV_Reader.cpp"
#include <time.h>

#define path "Datos\\gt_2015.csv"


double* getPCAAutos(double * dataRead, int m, int n) {
	// Constructor que prepara los datos para el cálculo del PCA (Centrado y covarianza)
	PCA_Autos pcaAutos(dataRead, m, n); 
	// Se calcula autovectores y autovalores, y se reordenan de mayor a menor (Matriz de covarianza)
	pcaAutos.autos('V', 'U'); 
	// Se obtiene los datos mediante PCA
	return pcaAutos.getPCA(); 
}

double* getPCASVD(double* dataRead, int m, int n) {
	// Constructor que prepara los datos para el cálculo del PCA (centrado y covarianza)
	PCA_SVD pcaSVD(dataRead, m, n);
	// Se calcula sus valores y vectores singulares
	pcaSVD.svd('A', 'A'); 
	// Se obtiene los datos mediante PCA
	return pcaSVD.getPCA(); 
}

int main(int argc, char* argv[]){
	CSV_Reader reader(path, ',');
	double* dataRead = reader.readData();	
	int m = reader.getM(); int n = reader.getN();

	double* pcaAutos = getPCAAutos(dataRead, m, n);
	double* pcaSVD = getPCASVD(dataRead, m, n);
	printf("Eig{24} = %f | SVD{24} = %f \n", pcaAutos[24], pcaSVD[24]);
				
	return 0;
}