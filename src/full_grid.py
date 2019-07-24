import numpy as np
from time import clock




def get_full_grid(dataset, edges_of_cell, edges_of_big_cell):
    """
    input: dataset ... numpy array, timestemp, position in space, measured values
           edges_of_cell ... list, lengths of individual edges of smallest cell
           edges_of_big_cell ... list, lower resolution cells (creates surroundings of measured data)
    output: domain_coordinates ... numpy array, "list" of coordinates of cells were some measurement was performed
            domain_frequencies ... numpy array, number of measurements in each cell
            domain_sums ... numpy array, sums of measured values
    """
    shape_of_grid, uniform_histogram_bools = get_uniform_data(dataset[:, 0: -1], edges_of_cell, edges_of_big_cell)
    domain_coordinates, shift, precision = get_coordinates(dataset[:, 0: -1], shape_of_grid, uniform_histogram_bools)
    domain_sums = get_sums(dataset, shape_of_grid, uniform_histogram_bools)
    #print(np.sum(dataset[:, -1]))
    return np.float64(domain_coordinates)/precision + shift, domain_sums





def number_of_cells(data, edges_of_cell):
    """
    input: X numpy array nxd, matrix of measures
           edge_of_square float, length of the edge of 2D part of a "cell"
           timestep float, length of the time edge of a "cell"
    output: shape_of_grid numpy array, number of edges on t, x, y, ... axis
    uses:np.shape(), np.max(), np.min(),np.ceil(), np.int64()
    objective: find out number of cells in every dimension
    """
    # number of predefined cubes in the measured space
    # changed to exact length of timestep and edge of square
    # changed to general shape of cell
    shape_of_grid = [[],[]]
    n, d = np.shape(data)
    for i in range(d):
        min_i = np.min(data[:, i])
        max_i = np.max(data[:, i])
        range_i = max_i - min_i
        edge_i = float(edges_of_cell[i])
        number_of_bins = np.floor(range_i / edge_i) + 1
        half_residue = (edge_i - (range_i % edge_i)) / 2.0
        position_min =  min_i - half_residue
        position_max =  max_i + half_residue
        shape_of_grid[0].append(int(number_of_bins))
        shape_of_grid[1].append([position_min, position_max])
    return shape_of_grid




def get_coordinates(data, shape_of_grid):
    """
    """
    edges = np.histogramdd(data, bins=shape_of_grid[0],
                                      range=shape_of_grid[1],
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


def get_sums(dataset, shape_of_grid):
    """
    """
    histogram = np.histogramdd(dataset[:, 0: -1], bins=shape_of_grid[0],
                                      range=shape_of_grid[1],
                                      normed=False, weights=dataset[:, -1])[0]
    histogram_sums = histogram.reshape(-1)
    return histogram_sums



