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

//using namespace std;

void readDataset(std::string file_name, std::vector<std::vector<double>> &vect);

void roundDataset(std::vector<std::vector<double>> &vect_list, std::vector<double> &roundings);

void applySet(std::vector<std::vector<double>> &vect_list, std::set<std::vector<double>> &rounded_set);

void countInSet(std::vector<std::vector<double>> &rounded_dataset, std::set<std::vector<double>> my_set);


#endif

