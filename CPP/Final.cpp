#include "CSVReader.cpp"
#include "PCA.cpp"
#include <cstdio>
#include <cstdlib>
#include<time.h>

#define path "Datos\\gt_2015.csv"

int main(int argc, char* argv[]) {
	// Read CSV file
	CSVReader reader(path, ',');	
	double* dataRead = reader.readData();	
	int m = reader.getM(); int n = reader.getN();
	double* dataPCA;

	clock_t start, stop;
	start = clock();
	for (int i = 0; i < 10000; i++) {
		// Apply pca 
		PCA pca(dataRead);
		pca.centerData(m, n);	// Matrix 7384x11
		pca.computeCov(m, n); // Matrix 7384x11
		// Se calcula autovectores y autovalores, y se reordenan de mayor a menor
		pca.autos('V', 'U', n, n);
		dataPCA = pca.getPCA(m, n); // PCA data
	}
	stop = clock();
	printf("Pos {11} = %f\n", dataPCA[11]);
	printf("Tiempo: %4.8f segundos\n", (double)(stop - start) / CLOCKS_PER_SEC / 10000);

		
	return 0;
}