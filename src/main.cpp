#include "load.h"
#include "grid.h"



int main(int argc, char *argv[]){

    // testing grid creation
	std::vector<std::vector<double>> dataset;
	readDataset("data.txt", dataset);
    std::cout << "delka datasetu: " << dataset.size() << std::endl;
    std::cout << "dimenze datasetu: " << dataset[0].size() << std::endl;
    int radius = 5;
    std::vector<double> cell_size{300.0, 0.05, 0.05};
	std::vector<std::vector<double>> grid;
    createGrid(dataset, grid, radius, cell_size);
    std::cout << "delka gridu: " << grid.size() << std::endl;
    std::cout << "dimenze gridu: " << grid[0].size() << std::endl;
	return 0;
}
