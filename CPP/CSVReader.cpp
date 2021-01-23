#include <string>
#include <fstream>
#include <sstream>
#include <vector>

using namespace std;

class CSVReader {
public: 
	CSVReader(string path_, char delim_) {
		path = path_;
		delim = delim_;
		m = 0;
		n = 0;
	}

	int getM() {
		return m / n;
	}

	int getN() {
		return n;
	}

	double* readData() {

		bool count = false;
		vector <string> vectData;

		ifstream fin(path);		
		
        string line, word, temp;

        while (getline(fin, line)) {

            istringstream s(line);
             
            while (getline(s, word, ',')) {
				if (!count) {
					n++;
					headers.push_back(word);
				} else {
					vectData.push_back(word);
				}				
            }
			count = true;
        }

		fin.close();

		return vectorToDouble(vectData);
	}

private:
	string path;
	vector <string> headers;
	char delim;
	int m, n;

	double * vectorToDouble(vector <string> vect) {
		m = vect.size(); 
		int cont = 0;
		double* data = new double[m];
		for (string x : vect) {
			data[cont] = stod(x);
			cont++;
		}
		return data;
	}
};