import directions
"""
**software used during developement:**
Ubuntu 18.04,
Python 2.7.15rc1,
Numpy 1.16.3,
sklearn 0.20.3,
SciPy 1.2.1,
dicttoxml 1.7.4
pandas 0.22.0

sudo apt update
sudo apt install python2.7 python-numpy python-sklearn python-scipy python-dicttoxml python-pandas

**run using:**
python run_directions.py
"""

# addresses must be defined, for example: '~/directory/file.txt'
training_dataset_address = None  # string, must be defined
address_of_xml_file = 'whyte_map.xml'  # string, can be defined

# parameters of the training dataset text file
delimiter = ' '  #  for csv probably ','
header = None  # with header containing strings header = 'infer'
usecols = (0, 2, 3, 5, 6)  # time, position x, position y, velocity, angle of motion

# parameters for the method
number_of_clusters = 3  # can be changed, must be at least 1
number_of_spatial_dimensions = 4  # do not change
list_of_periodicities = [86400.0]  # should be gathered from fremen

# derived parameter
structure_of_extended_space = [number_of_spatial_dimensions, list_of_periodicities]

# create the object
dirs = directions.Directions(clusters=number_of_clusters, structure=structure_of_extended_space)

# create the model
dirs = dirs.fit(training_dataset_address, delimiter, header, usecols)

# save the model to xml file
dirs.model2xml(address_of_xml_file)
