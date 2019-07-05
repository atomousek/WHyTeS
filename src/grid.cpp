#include "grid.h"




void expansion(int radius, int dim, std::vector<int> &line, std::vector<std::vector<int>> &output, int counter, int max_dim)
{
	if(dim > 0)
	{
		for(int i = -1*radius; i < radius+1; ++i )
		{
			line[dim] = i;
			expansion(radius, dim-1, line, output, counter+(pow(2*radius+1, dim))*(i+radius), max_dim);
		}
	}
	else
	{
		for(int i = -1*radius; i < radius + 1; ++i)
		{
			line[dim] = i;
			for(int j = 0; j < max_dim; ++j)
			{
				output[counter][j] = line[j];
			}
			++counter;
		}
	}
}

void expand(int radius, int dim, std::vector<std::vector<int>> &output)
{
	int counter = 0;
	//int length = pow((2*radius+1), dim);
	std::vector<int> line(dim);
	//vector<vector<int>> out (dim*length);
	expansion(radius, dim-1, line, output, counter, dim);
}


void extendGrid(std::set<std::vector<double>> &rounded_set, std::set<std::vector<double>> &extended_set, std::vector<std::vector<int>> &surroundings)
{
	std::vector<std::vector<double>> vect_list;	
	for(std::vector<double> vect : rounded_set)
	{
		for(std::vector<int> shift : surroundings)
		{

			std::vector<double> temp_vect;
	
			for(unsigned int i = 0; i < vect.size(); ++i)
			{
				temp_vect.push_back(vect[i]+shift[i]);
			}
			vect_list.push_back(temp_vect);
		}
	}
	extended_set.insert(vect_list.begin(), vect_list.begin());
}	

