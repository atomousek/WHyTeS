#ifndef __LOAD_H__
#define __LOAD_H__

#include <iostream>
#include <fstream>
#include <sstream>

#include <string>
#include <vector>
#include <set>

#include <cmath>


using namespace std;

void readDataset(string file_name, vector<vector<double>> &vect);

void roundDataset(vector<vector<double>> &vect_list, vector<double> &roundings, set<vector<double>> &rounded_set);

#endif

