
from dask_image.ndmeasure import label, area
import dask.array as da
import numpy as np

def remove_small_objects_dask(mask:da.array, min_size) -> da.array:
    """
    Remove small objects from a mask: compatible with dask arrays
    """
    # Get labels of mask
    labeled_zones, nb_zones = label(mask).astype(np.uint16)
    # Get area of each label
    areas = area(image=mask,label_image=labeled_zones, index=np.arange(1,nb_zones+1))

    # Get labels to remove
    if isinstance(min_size, str):
        if min_size == 'max':
            min_size = np.max(areas) * 0.95 # seems to make slight mistakes sometimes => 0.95 factor to compensate
        else:
            raise ValueError('min_size must be an integer or "max"')
    
    labels_to_remove = da.where(areas < min_size)[0] + 1
    # Remove labels
    labeled_zones[da.isin(labeled_zones, labels_to_remove)] = 0
    # Return image
    return labeled_zones.astype(bool)