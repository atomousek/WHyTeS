#ifndef __GRID_H__
#define __GRID_H__

#include <iostream>
#include <vector>
#include <cmath>
#include <set>
#include <unordered_set>

//using namespace std;
//
void expansion(int radius, int dim, std::vector<int> &line, std::vector<std::vector<int>> &output, int counter, int max_dim);

void expand(int radius, int dim, std::vector<std::vector<int>> &output);

void extendGrid(std::set<std::vector<double>> rounded_set, std::set<std::vector<double>> extended_set, std::vector<std::vector<int>> surroundings);


#endif
