from cdBoundary.boundary import ConcaveHull

import numpy as np

import shapely



def calc_concave_hull(geoseries):
    """
    Calculate the concave hull of polygons in a GeoPandas GeoSeries.

    Parameters:
    geoseries (geopandas.geoseries.GeoSeries): A GeoPandas GeoSeries containing polygons.

    Returns:
    shapely.geometry.Polygon: The concave hull of the polygons.

    Notes:
    Implementation for concave hull from:
    https://gist.github.com/AndreLester/589ea1eddd3a28d00f3d7e47bd9f28fb

    See dedicated page:
    https://github.com/civildot/cdBoundary
    """

    # Generate single points
    # Inner Helper Function that flattens a list
    def flatten_list(nested_list):
        return [item for sublist in nested_list for item in (flatten_list(sublist) if isinstance(sublist, list) else [sublist])]

    polygon_list = flatten_list([item if isinstance(item, shapely.Polygon) else list(item.geoms) for item in geoseries])
    point_list = np.array([np.array(coord) for polygon in polygon_list for coord in list(polygon.exterior.coords)])


    # Calculate Concave Hull
    ch = ConcaveHull() #class instance
    ch.loadpoints(point_list)
    ch.calculatehull(tol=ch.estimate()) #see documentation for tol argument

    boundary_points = np.vstack(ch.boundary_points()) # boundary_points is a subset of pts corresponding to the concave hull

    return shapely.Polygon(boundary_points)


def Weiszfield_geom_median(xy_array, tol=1e-6, max_iter=1000):
    """
    Calculate the geometric median of points using Weiszfield's algorithm (2-dimensional case).

    Parameters:
    xy_array (numpy.ndarray): A numpy array of shape (number_of_points, 2) containing points.
    tol (float, optional): Tolerance for convergence (default is 1e-6).
    max_iter (int, optional): Maximum number of iterations (default is 1000).

    Returns:
    shapely.geometry.Point: The geometric median of the points.

    Notes:
    Based on: https://gist.github.com/endolith/2837160

    But see https://doi.org/10.1145/2897518.2897647 for discussion and for linear time convergence in more dimensions
    """

    # Check if there's only one point, return its centroid
    if len(xy_array) == 1:
        return shapely.Point(xy_array)

    # Initialize
    test_median = np.mean(xy_array, axis=0)
    iter_count = 0
    diff = tol + 1
    diff_values = [diff]

    while iter_count < max_iter and diff > tol:
        # Calculate the distances and weights for all points
        distances = np.linalg.norm(xy_array - test_median, axis=1)
        weights = 1 / distances
        denom = np.sum(weights)

        # Calculate the next guess using weighted sum
        next_median = np.sum(xy_array * (weights / denom)[:, np.newaxis], axis=0)

        # Update guess
        diff = np.linalg.norm(test_median - next_median)
        iter_count += 1
        test_median = next_median

        # Append the current difference to the list
        diff_values.append(diff)


    # Plot the difference values
    #plt.figure()
    #plt.plot(range(iter_count+1), diff_values, marker='o',color='black')
    #plt.xlabel('Iterations')
    #plt.ylabel('Difference')
    #plt.yscale('log')
    #plt.title('Convergence of Weiszfield\'s Algorithm')
    #plt.grid(True)
    #plt.show()

    return shapely.Point(test_median)
