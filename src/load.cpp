#include "load.h"




void readDataset(std::string file_name, std::vector<std::vector<double>> &vect)
{
    /*
     	loads dataset from *.txt file
    	input: file name 
	output: loaded dataset as vector of vectors
    */
    
    std::ifstream file(file_name, std::ios::in);    // opens file 
    if (file.good())
    {
	std::string str;
        while(getline(file, str))    // takes line and saves it as string until there is end of line
        {
            std::vector<double> temp_vect;
	    std::istringstream ss(str);
            double num;
            while(ss >> num) 
            {
                temp_vect.push_back(num); // appends double to temp_vect
            }
            vect.push_back(temp_vect);  // appends temp_vect to vect
        }
    }
}

void roundDataset(std::vector<std::vector<double>> &vect_list, std::vector<double> &roundings)
{
	/*
	 	applies rounding function to datase
		input: dataset loaded as vector of vectors
		output: serial number of bins
	 
	 */

    for(std::vector<double> &vect : vect_list)  // for each vector<double> in vector<vector<double>>
    {
        for(unsigned int i = 0; i < vect.size(); ++i)
        {
           // vect[i] = floor(vect[i] / roundings[i]) * roundings[i] + (roundings[i] / 2.0);
	   vect[i] = (int)floor(vect[i] / roundings[i]);
        }
    }
   
}
 
void applySet(std::vector<std::vector<double>> &vect_list, std::set<std::vector<double>> &rounded_set)
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


