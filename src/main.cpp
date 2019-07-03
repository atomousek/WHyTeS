#include "load.h"
#include "grid.h"


int main(int argc, char *argv[]){

	vector<vector<double>> vect;
	readDataset("data.txt", vect);
	cout << vect[0][1] << endl;
	set<vector<double>> rounded_set;
	vector<double> roundings(3);
	roundings[0] = 0.5;
	roundings[1] = 0.2;
	roundings[2] = 0.2;
	roundDataset(vect, roundings);
	cout << "rounded " << vect.size() << endl;
	applySet(vect, rounded_set);
	cout << "set " << rounded_set.size() << endl;
	//countInSet(vect, rounded_set);
	return 0;
}
