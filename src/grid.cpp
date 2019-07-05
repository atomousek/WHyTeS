#include "grid.h"


void createGrid(std::vector<std::vector<double>> &dataset, std::vector<std::vector<double>> &grid, int radius, std::vector<double> cell_size)
{
    int dim = dataset[0].size();
    int height = pow(2*radius+1, dim);

    std::vector<std::vector<int>> surroundings(height);
    for(int i = 0; i < height; ++i)
    {
        surroundings[i].resize(dim);

    }
    std::vector<std::vector<int>> rounded_data;
    roundDataset(dataset, cell_size, rounded_data);

    std::set<std::vector<int>> rounded_set;
    applySet(rounded_data, rounded_set);
    
    expand(radius, dim, surroundings);

    std::unordered_set<std::vector<int>, VectorHash> extended_set(rounded_set.size()*surroundings.size());
    extendGrid(rounded_set, extended_set, surroundings);

    indexesToGrid(extended_set, grid, cell_size);

    
} 

void indexesToGrid(std::unordered_set<std::vector<int>, VectorHash> extended_set, std::vector<std::vector<double>> &grid, std::vector<double> cell_size) 
{
    std::copy(extended_set.begin(), extended_set.end(), grid.begin());
    
    for(std::vector<double> &vect : grid)
    {
        for(unsigned int i = 0; i < cell_size.size(); ++i)
        {
            vect[i] = vect[i]*cell_size[i] + cell_size[i]/2.0;
        }
    }

}

void roundDataset(std::vector<std::vector<double>> &vect_list, std::vector<double> &cel_size, std::vector<std::vector<int>>  &int_vector)
{

    for(std::vector<double> &vect : vect_list)  // for each vector<double> in vector<vector<double>>
    {
	std::vector<int> temp_vect;
        for(unsigned int i = 0; i < vect.size(); ++i)
        {
           // vect[i] = floor(vect[i] / cel_size[i]) * cel_size[i] + (cel_size[i] / 2.0);
            temp_vect.push_back(floor(vect[i] / cel_size[i]));
        }
	int_vector.push_back(temp_vect);
    }

}


void applySet(std::vector<std::vector<int>> &vect_list, std::set<std::vector<int>> &rounded_set)
{
        /*
                makes set of vectors out of vector of vectors
                input: vector of vectors (vect_list)
                output: set of vectors (rounded_set)
         */

        rounded_set.insert(vect_list.begin(), vect_list.end());   // makes set out of vector<vector<double>>
}


void countInSet(std::vector<std::vector<double>> &rounded_dataset, std::set<std::vector<double>> my_set)
{
        int num;
        std::vector<std::vector<double>> vect_from_set(my_set.size());
        copy(my_set.begin(), my_set.end(), vect_from_set.begin());
        std::cout << "copied " << vect_from_set.size() << std::endl;
        for(std::vector<double> &vect : vect_from_set)
        {
                num = count(rounded_dataset.begin(), rounded_dataset.end(), vect);
                vect.push_back(num);
                //cout << num << endl;
        }
}



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


void extendGrid(std::set<std::vector<int>> &rounded_set, std::unordered_set<std::vector<int>, VectorHash> &extended_set, std::vector<std::vector<int>> &surroundings)
{
	//std::vector<std::vector<int>> vect_list;	
	for(const std::vector<int> &vect : rounded_set)
	{
		for(const std::vector<int> &shift : surroundings)
		{

			std::vector<int> temp_vect;
	
			for(unsigned int i = 0; i < vect.size(); ++i)
			{
				temp_vect.push_back(vect[i]+shift[i]);
			}
			extended_set.insert(temp_vect);
		}
	}
    //std::cout << "number of vectors before applying set " << vect_list.size() << std::endl;
	//std::set<std::vector<int>> my_set(vect_list.begin(), vect_list.end());
    //extended_set.insert(vect_list.begin(), vect_list.end());
    //std::cout << "number of vectors after applying set " << extended_set.size() << std::endl;

}	

