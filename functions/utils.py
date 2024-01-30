
from dask_image.ndmeasure import label, area
import dask.array as da
import numpy as np

def remove_small_objects_dask(mask:da.array, min_size:int) -> da.array:
    """
    Remove small objects from a mask: compatible with dask arrays
    """
    # Get labels of mask
    labeled_zones, nb_zones = label(mask)
    # Get area of each label
    areas = area(image=mask,label_image=labeled_zones, index=np.arange(1,nb_zones+1))
    # Get labels to remove
    labels_to_remove = da.where(areas < min_size)[0] + 1
    # Remove labels
    labeled_zones[da.isin(labeled_zones, labels_to_remove)] = 0
    # Return image
    return labeled_zones.astype(bool)