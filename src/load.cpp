#include "load.h"

int main(int argc, char *argv[])
{

    vector<vector<double>> vect;
    set<vector<double>> rounded_set;
    readDataset("data.txt", vect);
    cout << vect[0][0] << endl;
    
    return 0;
}

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

void roundDataset(vector<vector<double>> &vect_list, vector<double> &roundings, set<vector<double>> &rounded_set)
{
    for(auto &vect : vect_list)  // for each vector<double> in vector<vector<double>>
    {
        for(unsigned int i = 0; i < vect.size(); ++i)
        {
            vect[i] = floor(vect[i] / roundings[i]) * roundings[i] + (roundings[i] / 2.0);
        
        }
    }
    rounded_set.insert(vect_list.begin(), vect_list.end());   // makes set out of vector<vector<double>>
}
 

