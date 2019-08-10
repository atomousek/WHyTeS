# clone branch to separate directory

git clone -b 2019_ecmr --single-branch https://github.com/atomousek/WHyTeS

# Pedestrian flows modelling using WHyTe and testing using Breadth first search

In progress. Not functional.

This repo contains datasets and experiments related to the RAL (icra2020).
The readme will be extended based on the RAL reviews.


**software used during developement:**
Ubuntu 18.04,
Python 2.7.15rc1,
Numpy 1.16.3,
sklearn 0.20.3,
SciPy 1.2.1,
Cython 0.29.12
gsl 2.4
CythonGSL 0.2.2

**before you run the method:**
if you want to run with different cell size please change the following variables; 
* 'edges_of_cell' in run_method.py, test.py and make_video.py
* 'time_interval', 'x_interval', 'y_interval' at the end of slider.py

**example, how to run method:**
- $ python run_method.py
- $ python slider.py
- $ python test.py
- $ python make_video.py


**format of data:**
text files with measurements in rows and variables in columns ordered in this way:

timestamp,
spatial variables. (not finished)

