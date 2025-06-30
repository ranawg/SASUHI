from pyproj import Transformer

def project_bbox_to_target_bounds(bbox_wgs84, target_crs="EPSG:32637", resolution=1000):
    """
    Projects a WGS84 bounding box to the target CRS and snaps to grid resolution.
    
    Parameters:
        bbox_wgs84 (tuple): (minlon, minlat, maxlon, maxlat)
        target_crs (str): EPSG code for target CRS
        resolution (int): grid resolution in target CRS units (e.g., meters)
    
    Returns:
        tuple: (xmin, ymin, xmax, ymax) in projected coordinates
    """
    transformer = Transformer.from_crs("EPSG:4326", target_crs, always_xy=True)
    minlon, minlat, maxlon, maxlat = bbox_wgs84
    xmin, ymin = transformer.transform(minlon, minlat)
    xmax, ymax = transformer.transform(maxlon, maxlat)

    # Snap to nearest resolution step
    xmin = int(xmin // resolution) * resolution
    xmax = int(np.ceil(xmax / resolution)) * resolution
    ymin = int(ymin // resolution) * resolution
    ymax = int(np.ceil(ymax / resolution)) * resolution

    return (xmin, ymin, xmax, ymax)


def add_buffer(bbox, buffer_deg):
    minlon, minlat, maxlon, maxlat = bbox
    return (
        minlon - buffer_deg,
        minlat - buffer_deg,
        maxlon + buffer_deg,
        maxlat + buffer_deg
    )