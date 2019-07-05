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


