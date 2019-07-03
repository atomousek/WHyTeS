#include "load.h"
/*
int main(int argc, char *argv[])
{

    vector<vector<double>> vect;
    set<vector<double>> rounded_set;
    readDataset("data.txt", vect);
    cout << vect[0][0] << endl;
    
    return 0;
}*/

void readDataset(string file_name, vector<vector<double>> &vect)
{
    /*
     
    */
    
    ifstream file(file_name, ios::in);    // opens file 
    if (file.good())
    {
        string str;
        while(getline(file, str))    // takes line and saves it as string until there is end of line
        {
            vector<double> temp_vect;
            istringstream ss(str);
            double num;
            while(ss >> num) 
            {
                temp_vect.push_back(num); // appends double to temp_vect
            }
            vect.push_back(temp_vect);  // appends temp_vect to vect
        }
    }
}

void roundDataset(vector<vector<double>> &vect_list, vector<double> &roundings)
{
    for(auto &vect : vect_list)  // for each vector<double> in vector<vector<double>>
    {
        for(unsigned int i = 0; i < vect.size(); ++i)
        {
            vect[i] = floor(vect[i] / roundings[i]) * roundings[i] + (roundings[i] / 2.0);
        
        }
    }
   
}
 
void applySet(vector<vector<double>> &vect_list, set<vector<double>> &rounded_set){
	rounded_set.insert(vect_list.begin(), vect_list.end());   // makes set out of vector<vector<double>>
}


void countInSet(vector<vector<double>> &rounded_dataset, set<vector<double>> my_set)
{
	int num;
	vector<vector<double>> vect_from_set(my_set.size());
	copy(my_set.begin(), my_set.end(), vect_from_set.begin());
	cout << "copied " << vect_from_set.size() << endl;
	for(auto &vect : vect_from_set)
	{
		num = count(rounded_dataset.begin(), rounded_dataset.end(), vect);
		vect.push_back(num);
		//cout << num << endl;
	}	
}


