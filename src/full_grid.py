import numpy as np
from time import clock




def get_full_grid(dataset, edges_of_cell):
    """
    input: dataset ... numpy array, timestemp, position in space, measured values
           edges_of_cell ... list, lengths of individual edges of smallest cell
           edges_of_big_cell ... list, lower resolution cells (creates surroundings of measured data)
    output: coordinates ... numpy array, "list" of coordinates of cells were some measurement was performed
            domain_frequencies ... numpy array, number of measurements in each cell
            k_for_poisson ... numpy array, sums of measured values
    """
    bins_and_ranges = get_bins_and_ranges(dataset[:, 0: -1], edges_of_cell)
    #coordinates, shift, precision = get_coordinates(dataset[:, 0: -1], bins_and_ranges)
    coordinates = get_coordinates_no_optimization(dataset[:, 0: -1], bins_and_ranges, edges_of_cell)
    frequencies = get_frequencies(dataset, bins_and_ranges)
    #return np.float64(coordinates)/precision + shift, frequencies
    return coordinates, frequencies





def get_bins_and_ranges(data, edges_of_cell):
    """
    input: X numpy array nxd, matrix of measures
           edge_of_square float, length of the edge of 2D part of a "cell"
           timestep float, length of the time edge of a "cell"
    output: bins_and_ranges numpy array, number of edges on t, x, y, ... axis
    uses:np.shape(), np.max(), np.min(),np.ceil(), np.int64()
    objective: find out number of cells in every dimension
    """
    # number of predefined cubes in the measured space
    # changed to exact length of timestep and edge of square
    # changed to general shape of cell
    bins_and_ranges = [[],[]]
    n, d = np.shape(data)
    #print(np.shape(data))
    for i in range(d):
        min_i = np.min(data[:, i])
        max_i = np.max(data[:, i])
        range_i = max_i - min_i
        edge_i = float(edges_of_cell[i])
        #edge_i = edges_of_cell[i]
        number_of_bins = np.floor(range_i / edge_i) + 1
        half_residue = (edge_i - (range_i % edge_i)) / 2.0
        position_min =  min_i - half_residue
        position_max =  max_i + half_residue
        #print(type(position_max))
        #print(type(max_i))
        #print(type(half_residue))
        #print(type(range_i))
        #print(type(edge_i))
        bins_and_ranges[0].append(int(number_of_bins))
        bins_and_ranges[1].append([position_min, position_max])
        #print(type(bins_and_ranges[1][0][0]))
    return bins_and_ranges


def get_coordinates_no_optimization(data, bins_and_ranges, edges_of_cell):
    edges = np.histogramdd(data, bins=bins_and_ranges[0],
                                      range=bins_and_ranges[1],
                                      normed=False, weights=None)[1]
    central_points = []
    for i in range(len(edges)):
        central_points.append(np.array(edges[i][0: -1]) + edges_of_cell[i]/2.0)
    return cartesian_product(*central_points)




def get_coordinates(data, bins_and_ranges):
    """
    """
    edges = np.histogramdd(data, bins=bins_and_ranges[0],
                                      range=bins_and_ranges[1],
                                      normed=False, weights=None)[1]
    central_points = []
    shift = []
    precision = []
    for i in range(len(edges)):
        step_lenght = (edges[i][-1] - edges[i][0]) / (len(edges[i] - 1))
        if i == 0:
            prec = 100.0
            tmp = np.uint32((np.array(edges[i][0: -1]) - edges[i][0]) * prec)
            shift.append(edges[i][0] + step_lenght / 2.0)
            #tmp = np.array(edges[i][0: -1] + step_lenght / 2.0, dtype=np.uint32)
            #shift.append(0.0)
            precision.append(prec)
        else:
            prec = 100000.0
            tmp = np.uint32((np.array(edges[i][0: -1]) - edges[i][0]) * prec)
            shift.append(edges[i][0] + step_lenght / 2.0)
            precision.append(prec)
        central_points.append(tmp)
    shift = np.array(shift)
    precision = np.array(precision)
    coordinates = cartesian_product(*central_points)
    return coordinates, shift, precision


def get_frequencies(dataset, bins_and_ranges):
    """
    """
    frequencies = np.histogramdd(dataset[:, 0: -1], bins=bins_and_ranges[0],
                                      range=bins_and_ranges[1],
                                      normed=False, weights=dataset[:, -1])[0]
    return frequencies.reshape(-1)




def cartesian_product(*arrays):
    """
    downloaded from:
    'https://stackoverflow.com/questions/11144513/numpy-cartesian-product-of'+\
    '-x-and-y-array-points-into-single-array-of-2d-points'
    input: *arrays enumeration of central_points
    output: numpy array (central positions of cels of grid)
    uses: np.empty(),np.ix_(), np.reshape()
    objective: to perform cartesian product of values in columns
    """
    la = len(arrays)
    arr = np.empty([len(a) for a in arrays] + [la],
                   dtype=arrays[0].dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[..., i] = a
    return arr.reshape(-1, la)
