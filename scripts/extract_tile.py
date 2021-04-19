# --------------------------------------------------------------------------
# Extract data tile from given label region. Leverages coordinates from each
# raster to match the region of interest.
# --------------------------------------------------------------------------
import re
import argparse
import xarray as xr
import rasterio as rio
from shapely.geometry import box
import geopandas as gpd
from rasterio.mask import mask

__author__ = "Jordan A Caraballo-Vega, Science Data Processing Branch"
__email__ = "jordan.a.caraballo-vega@nasa.gov"
__status__ = "Production"


def arg_parser():
    """
    Argparser function
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-l', action='store', dest='label_filename', type=str,
        help='label raster filename'
    )
    parser.add_argument(
        '-i', action='store', dest='image_filename', type=str,
        help='image raster filename'
    )
    parser.add_argument(
        '-o', action='store', dest='out_filename', type=str,
        help='output image raster filename'
    )
    return parser.parse_args()


def getFeatures(gdf):
    """
    Function to parse features from GeoDataFrame to input rasterio
    """
    import json
    return [json.loads(gdf.to_json())['features'][0]['geometry']]


def gen_box_coordinates(small_image, epsg_code):
    """
    Generate study area bounding box
    """
    # min and max values on x
    minx = small_image.coords['x'][0].values
    maxx = small_image.coords['x'][small_image.shape[2] - 1].values

    # min and max values on y
    miny = small_image.coords['y'][small_image.shape[1] - 1].values
    maxy = small_image.coords['y'][0].values

    bbox = box(minx, miny, maxx, maxy)  # generate bounding box
    geop = gpd.GeoDataFrame(
        {'geometry': bbox}, index=[0], crs=epsg_code
    )
    return getFeatures(geop)  # return coordinates in polygon position


# -------------------------------------------------------------------------------
# main
# -------------------------------------------------------------------------------

if __name__ == "__main__":

    # ---------------------------------------------------------------------------
    # Initialize args parser
    # ---------------------------------------------------------------------------
    args = arg_parser()

    # read input data
    label_data = xr.open_rasterio(args.label_filename)
    image_data = rio.open(args.image_filename)

    epsg_code = int(re.sub(r'^.*?:', '', label_data.crs))
    tile_coords = gen_box_coordinates(label_data, epsg_code)

    out_img, out_transform = mask(
        dataset=image_data, shapes=tile_coords, crop=True
    )
    out_meta = image_data.meta.copy()

    out_meta.update(
        {
            "driver": "GTiff",
            "height": out_img.shape[1],
            "width": out_img.shape[2],
            "transform": out_transform,
            "crs": epsg_code
        }
    )

    # output imagery raster clip
    with rio.open(args.out_filename, "w", **out_meta) as dest:
        dest.write(out_img)
