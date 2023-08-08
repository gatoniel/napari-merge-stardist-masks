"""
This module is an example of a barebones sample data provider for napari.

It implements the "sample data" specification.
see: https://napari.org/stable/plugins/guides.html?#sample-data

Replace code below according to your needs.
"""
from __future__ import annotations

from importlib_resources import as_file, files
from tifffile import imread


def stardist_opp_sample_data():
    """Generates an image"""
    with as_file(
        files("napari_merge_stardist_masks.data").joinpath(
            "biofilm_1_cropped_again_raw.tif"
        )
    ) as f:
        img = imread(f)

    # original size (zyx = 129x129x129) is problematic
    img = img[:, :128, :128]

    return [(img, dict(name="StarDist OPP sample data"))]
