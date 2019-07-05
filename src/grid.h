#ifndef __GRID_H__
#define __GRID_H__


#include <iostream>
#include <fstream>
#include <sstream>

#include <string>
#include <vector>
#include <set>
#include <unordered_set>

#include <cmath>
#include <algorithm>


//using namespace std;

struct VectorHash {
    size_t operator()(const std::vector<int>& v) const {
        std::hash<int> hasher;
        size_t seed = 0;
        for (int i : v) {
            seed ^= hasher(i) + 0x9e3779b9 + (seed<<6) + (seed>>2);
        }
        return seed;
    }
};



void roundDataset(std::vector<std::vector<double>> &vect_list, std::vector<double> &roundings, std::vector<std::vector<int>> &int_vector);

void applySet(std::vector<std::vector<int>> &vect_list, std::set<std::vector<int>> &rounded_set);

void countInSet(std::vector<std::vector<double>> &rounded_dataset, std::set<std::vector<double>> my_set);

void expansion(int radius, int dim, std::vector<int> &line, std::vector<std::vector<int>> &output, int counter, int max_dim);

void expand(int radius, int dim, std::vector<std::vector<int>> &output);

void extendGrid(std::set<std::vector<int>> &rounded_set, std::unordered_set<std::vector<int>, VectorHash> &extended_set, std::vector<std::vector<int>> &surroundings);

void createGrid(std::vector<std::vector<double>> &dataset, std::vector<std::vector<double>> &grid, int radius, std::vector<double> cell_size);


#endif
