#include "grid.h"

void expansion(int radius, int dim, std::vector<int> &line, std::vector<vector<int>> &out , int counter ,int max_dim)
{
	if(dim > 0)
	{
		for(int i = -1*radius; i < radius+1; ++i )
		{
			line[dim] = i;
		}
		expansion(radius, dim-1, line, out, counter+((2*radius+1)**dim)*(i+radius), max_dim);
	}
	else
	{
		for(int i = -1*radius; i < radius + 1; ++i)
		{
			line[dim] = i;
			for(int j = 0; j < max_dim; ++j
			{
				out[counter][j] = line[j];
			}
			++counter;
		}
	
	}


}

