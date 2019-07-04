#include "grid.h"
/*
int main(){

}
*/
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

