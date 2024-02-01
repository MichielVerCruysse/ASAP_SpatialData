from scipy import ndimage
from scipy.spatial import distance
import numpy as np


def frame_around_object(prop, size_increase: int, im_dimensions: tuple) -> tuple:
    """
    Calculate edges of frame around input object
    :param prop: skimage.regionprops property of current object around which a box should be drawn
    :param size_increase: nb of pixels with which the frame should expand
    :param im_dimensions: dimensions of image from which object originates (used to check of new frame does not
                          cross image borders
    :return: corner points xs and ys of expanded frame: (y1, y2, x1, x2)
    """

    y_max, x_max = im_dimensions
    y1, x1, y2, x2 = prop.bbox
    frame_y1, frame_x1 = y1 - size_increase,  x1 - size_increase
    frame_y2, frame_x2 = y2 + size_increase,  x2 + size_increase
    return max(0, frame_y1), min(y_max, frame_y2), max(0, frame_x1), min(x_max, frame_x2)






def get_shortest_axis(object1_coords: np.array, object2_coords: np.array) -> tuple:
    """
    Calculates the shortest distance and corresponding points (p1, p2) from the coordinate lists of both objects with
    pi of form (yi, xi)
    :return: ((p1, p2), minimal distance)
    """

    # calculate distance matrix between c and p bounds
    dist_matrix = distance.cdist(object1_coords, object2_coords)
    min_distance = np.min(dist_matrix)

    # find corresponding c and p coords
    c_index, p_index = np.where(dist_matrix == min_distance)

    central_point, portal_point = object1_coords[c_index], object2_coords[p_index]
    axis = (central_point[0], portal_point[0])

    return axis, min_distance