# LOAM - poLygon extractiOn from rAster Map

LOAM: Exploiting Polygon Metadata to Understand Raster Maps - Accurate Polygonal Feature Extraction

A two-stage approach that exploits the polygon metadata to extract geological features from raster maps.



## Abstract

Locating undiscovered deposits of critical minerals requires accurate geological data. However, most of the 100,000 historical geological maps of the United States Geological Survey (USGS) are in raster format. This hinders critical mineral assessment. We target the problem of extracting geological features represented as polygons from raster maps. We exploit the polygon metadata that provides information on the geological features, such as the map keys indicating how the polygon features are represented, to extract the features. We present a metadata-driven machine-learning approach that encodes the raster map and map key into a series of bitmaps and uses a convolutional model to learn to recognize the polygon features. We evaluated our approach on USGS geological maps; our approach achieves a median F1 score of 0.809 and outperforms state-of-the-art methods by 4.52\%.



## Introduction





## Environment

### Create from Conda Config

```
conda env create -f environment.yml
conda activate loam
```

### Create from Separate Steps
```
conda create -n loam python=3.9.16
conda activate loam

conda install pytorch torchvision==1.13.1 torchaudio==0.13.1 cudatoolkit=11.7 -c pytorch
pip install -r requirements.txt
```


## Usage

Run .ipynb



## Dataset

[DARPA Critical Mineral Assessment Competition - Map Feature Extraction Challenge](https://criticalminerals.darpa.mil/The-Competition)

```
@dataset{cma2023data,
  author    = {Goldman, M.A. and Rosera, J.M. and Lederer, G.W. and Graham, G.E. and Mishra, A. and Yepremyan, A.},
  title     = {Training and validation data from the AI for Critical Mineral Assessment Competition},
  year      = {2023},
  doi       = {10.5066/P9FXSPT1},
  publisher = {U.S. Geological Survey data release},
}
```

[Competition Leaderboard (October 2022)](https://web.archive.org/web/20221202080740/https://criticalminerals.darpa.mil/Leaderboard)



## Citation

Details to be determined

```
@InProceedings{,
  title     = {Exploiting Polygon Metadata to Understand Raster Maps: Accurate Polygonal Feature Extraction},
  author    = {Lin, Fandel and Knoblock, Craig A and Shbita, Basel and Vu, Binh and Li, Zekun and Chiang, Yao-Yi},
  booktitle = {Proceedings of the 31st ACM International Conference on Advances in Geographic Information Systems (SIGSPATIAL '23)},
  pages     = {},
  year      = {2023},
  doi       = {10.1145/3589132.3625659}
}
```
