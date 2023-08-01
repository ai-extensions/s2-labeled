import os
from osgeo import gdal, ogr, osr
from sklearn.model_selection import train_test_split
import numpy as np

gdal.UseExceptions()


def pixel_to_coords(source, x, y):
    """Returns global coordinates in EPSG:4326 from pixel x, y coords"""

    geo_transform = source.GetGeoTransform()

    x_min = geo_transform[0]
    x_size = geo_transform[1]
    y_min = geo_transform[3]
    y_size = geo_transform[5]
    px = x * x_size + x_min
    py = y * y_size + y_min

    srs = osr.SpatialReference()
    srs.ImportFromWkt(source.GetProjection())

    srs_4326 = srs.CloneGeogCS()
    ct = osr.CoordinateTransformation(srs, srs_4326)

    long, lat, _ = ct.TransformPoint(px, py)

    return lat, long


def to_geojson(t, x, y):
    """Converts the given x, y, and split dataset type (train, test, validate ) to a geojson file
    The geojson file is saved in the current directory with the name label-{t}.geojson
    """

    field_name = "class"
    field_type = ogr.OFTString

    # Create the output Driver
    out_driver = ogr.GetDriverByName("GeoJSON")

    geojson_filename = f"label-{t}.geojson"
    # Create the output GeoJSON
    out_datasource = out_driver.CreateDataSource(geojson_filename)
    out_layer = out_datasource.CreateLayer("labels", geom_type=ogr.wkbPolygon)
    id_field = ogr.FieldDefn(field_name, field_type)
    out_layer.CreateField(id_field)
    # Get the output Layer's Feature Definition
    feature_def = out_layer.GetLayerDefn()

    for index, v in enumerate(y):
        point = ogr.Geometry(ogr.wkbPoint)
        point.AddPoint(x[index][0], x[index][1])

        # create a new feature
        out_feature = ogr.Feature(feature_def)

        # Set new geometry
        out_feature.SetGeometry(point)

        out_feature.SetField(field_name, v)
        # Add new feature to output Layer
        out_layer.CreateFeature(out_feature)

        # dereference the feature
        out_feature = None

    # Save and close DataSources
    out_datasource = None
