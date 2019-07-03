#ifndef __LOAD_H__
#define __LOAD_H__

#include <iostream>
#include <fstream>
#include <sstream>

#include <string>
#include <vector>
#include <set>

#include <cmath>
#include <algorithm>

using namespace std;

void readDataset(string file_name, vector<vector<double>> &vect);

void roundDataset(vector<vector<double>> &vect_list, vector<double> &roundings);

void applySet(vector<vector<double>> &vect_list, set<vector<double>> &rounded_set);

void countInSet(vector<vector<double>> &rounded_dataset, set<vector<double>> my_set);


#endif

