from dask_image.ndmorph import binary_closing, binary_opening
import dask.array as da
import numpy as np
from utils import remove_small_objects_dask

# zone segmentation functions

def segment_central_zones(glul_channel:da.array, structuring_element:np.array, central_quantile_threshold:int = 85 ,central_size_threshold:int=5000) -> da.array:

    # Assess quantile-intensity-based threshold 

    # Channel was masked => tissue = non-zero values
    tissue_intensities = glul_channel[glul_channel > 0]
    # Get threshold value: 85th percentile value
    threshing_value = np.percentile(tissue_intensities, central_quantile_threshold)
    raw_central_zone_segments = glul_channel > threshing_value 

    # Clean up central zone segments

    # Morph open
    opened_central_zone_segments = binary_opening(raw_central_zone_segments, structure=structuring_element) 
    # Remove small objects
    central_veins = remove_small_objects_dask(opened_central_zone_segments, min_size=central_size_threshold) 

    return da.asarray(central_veins, dtype=bool)


def segment_periportal_zones(ecadh_channel:da.array, structuring_element:np.array,  portal_quantile_threshold:int=62 ,small_portal_size_threshold:int=500, big_portal_size_threshold:int=5000) -> da.array:

    # Get threshold value: 62nd percentile value
    tissue_intensities = ecadh_channel[ecadh_channel > 0]
    threshing_value = np.percentile(tissue_intensities, portal_quantile_threshold)
    raw_periportal_zone_segments = ecadh_channel > threshing_value

    # Clean up periportal zones
    raw_portal_zones = remove_small_objects_dask(raw_periportal_zone_segments min_size=small_portal_size_threshold) 
    opened_periportal_zone_segments = binary_closing(da.asarray(raw_portal_zones), structure=structuring_element) 
    periportal_zone_segments = remove_small_objects_dask(opened_periportal_zone_segments, min_size=big_portal_size_threshold)

    return da.asarray(periportal_zone_segments)






