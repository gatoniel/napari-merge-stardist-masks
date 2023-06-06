"""
This module is an example of a barebones QWidget plugin for napari

It implements the Widget specification.
see: https://napari.org/stable/plugins/guides.html?#widgets

Replace code below according to your needs.
"""
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from magicgui import magicgui
from merge_stardist_masks.naive_fusion import naive_fusion
from merge_stardist_masks.utils import grid_from_path, rays_from_path
from napari.layers import Labels
from qtpy.QtWidgets import QWidget

if TYPE_CHECKING:
    import napari


@magicgui
def run_naive_fusion(
    model_path: Path,
    dists: "napari.types.ImageData",
    probs: "napari.types.ImageData",
    time: bool = False,
    subtract_dist: float = 0.5,
    prob_thresh: float = 0.65,
    no_slicing: bool = False,
    max_full_overlaps: int = 50,
    erase_probs_at_full_overlap: bool = False,
    show_overlaps: bool = False,
) -> "napari.layers.Labels":
    grid = grid_from_path(str(model_path))
    if time:
        lbls = []
        if probs.ndim == 4:
            rays = rays_from_path(str(model_path))
            _transpose = (1, 2, 3, 0)
        else:
            rays = None
            _transpose = (1, 2, 0)
        for i in range(dists.shape[1]):
            lbls.append(
                naive_fusion(
                    dists[:, i, ...].transpose(*_transpose) - subtract_dist,
                    probs[i, ...],
                    rays,
                    prob_thresh,
                    grid=grid,
                    max_full_overlaps=max_full_overlaps,
                    no_slicing=no_slicing,
                    show_overlaps=show_overlaps,
                    erase_probs_at_full_overlap=erase_probs_at_full_overlap,
                )
            )
        lbl = np.stack(lbls, axis=0)
    else:
        if probs.ndim == 3:
            rays = rays_from_path(str(model_path))
            _transpose = (1, 2, 3, 0)
        else:
            rays = None
            _transpose = (1, 2, 0)
        lbl = naive_fusion(
            dists.transpose(*_transpose) - subtract_dist,
            probs,
            rays,
            prob_thresh,
            grid=grid,
            max_full_overlaps=max_full_overlaps,
            no_slicing=no_slicing,
            show_overlaps=show_overlaps,
            erase_probs_at_full_overlap=erase_probs_at_full_overlap,
        )
    return Labels(lbl, name="StarDist OPP")


class StarDistOPPWidget(QWidget):
    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer

        _, w = self.viewer.window.add_plugin_dock_widget(
            "stardist-napari", "StarDist"
        )
        w.prob_thresh.value = 1.0
        w.cnn_output.value = True
        w.output_type.value = "Label Image"

        self.viewer.window.add_dock_widget(
            run_naive_fusion,
        )
        # self.setLayout(QHBoxLayout())
        # self.layout().addWidget(run_naive_fusion)
