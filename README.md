# LOAM
LOAM: Exploiting Polygon Metadata to Understand Raster Maps - Accurate Polygonal Feature Extraction

A two-stage approach that exploits the metadata to extract geological features from raster maps.

## Abstract

Locating undiscovered deposits of critical minerals requires accurate geological data. However, most of the 100,000 historical geological maps of the United States Geological Survey (USGS) are in raster format. This hinders critical mineral assessment. We target the problem of extracting geological features represented as polygons from raster maps. We exploit the polygon metadata that provides information on the geological features, such as the map keys indicating how the polygon features are represented, to extract the features. We propose a metadata-driven machine-learning approach that encodes the raster map and map key into a series of bitmaps and uses a convolutional model to learn to recognize the polygon features. We evaluate our approach on the USGS geological maps; our approach achieves a 0.809 median F1 score and outperforms state-of-the-art methods by 4.52%.

## Usage

