from scipy.ndimage import binary_fill_holes
from skimage.measure import label, regionprops
import dask.array as da 
from skimage.morphology import remove_small_objects
import nimpy as np
import cv2

def minmax_rescaling(channel: da.array, feature_range:tuple=(0, 1), dtype=np.float32) -> da.array:

    # Convert to float32 so we can divide by max value without losing information
    channel = channel.astype(np.float32)

    # Rescale image to 0-1 range
    min_value, max_value = da.min(channel), da.max(channel)
    channel = (channel - min_value) / (max_value - min_value)

    # Rescale image to feature range
    channel = channel * (feature_range[1] - feature_range[0]) + feature_range[0]

    return channel.astype(dtype)



def get_cut_mask(dapi_channel: da.array, median_value: int, threshold_quartile:int = 70) -> da.array:
    """Get cut mask from dapi channel by thresholding and removing small objects"""

    # Assess intensity threshold value
    q65 = np.percentile(dapi_channel[dapi_channel > median_value].flatten(), threshold_quartile)

    # Threshold and clean dapi channel
    threshed_dapi = dapi_channel > q65
    threshed_dapi = cv2.morphologyEx(threshed_dapi.astype(np.uint8).compute(), cv2.MORPH_CLOSE, np.ones((30, 30), np.uint8)) # TODO fix full read!!!!
    threshed_dapi = binary_fill_holes(threshed_dapi)

    # get area of largest object
    segment_props = regionprops(label(threshed_dapi))
    max_area = np.max([prop.area for prop in segment_props])

    # allow only largest segment
    cut_segment = remove_small_objects(threshed_dapi, min_size=max_area * 0.9) # regionprops.area seems to make slight mistakes sometimes => 0.9 scalar to compensate

    return da.asarray(cut_segment)

