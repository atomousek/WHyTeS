#include "load.h"
#include "grid.h"


int main(int argc, char *argv[]){

	std::vector<std::vector<double>> vect;
	readDataset("data.txt", vect);
	std::cout << vect[0][1] << std::endl;
	std::set<std::vector<double>> rounded_set;
	std::vector<double> roundings(3);

	roundings[0] = 600.0;
	roundings[1] = 0.2;
	roundings[2] = 0.2;

	int dim = 3;
	int radius = 5;
	int height = pow(2*radius+1, dim);

	std::vector<std::vector<int>> matrix(height);
	for(int i = 0; i < height; ++i)
	{
		matrix[i].resize(dim);
	
	}

	roundDataset(vect, roundings);
	std::cout << "rounded " << vect.size() << std::endl;
	applySet(vect, rounded_set);
	expand(radius, dim, matrix);

	/* 
	for(int i = 0; i < height; ++i)
	{
		for(int j = 0; j < dim; ++j)
		{

			//cout << matrix[i][j] << " ";
			printf("%2d ", matrix[i][j]);
		}
		printf("\n");	
		//cout << endl;
	}*/

	std::cout << "set " << rounded_set.size() << std::endl;

	std::set<std::vector<double>> extended_set;
	extendGrid(rounded_set, extended_set, matrix);

	//countInSet(vect, rounded_set);
	return 0;
}
