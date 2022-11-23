"""
This module is an example of a barebones QWidget plugin for napari

It implements the Widget specification.
see: https://napari.org/stable/plugins/guides.html?#widgets

Replace code below according to your needs.
"""
from typing import TYPE_CHECKING

import numpy as np
from magicgui import magicgui
from merge_stardist_masks.naive_fusion import naive_fusion_isotropic_grid
from napari.layers import Labels
from qtpy.QtWidgets import QWidget

if TYPE_CHECKING:
    import napari


@magicgui
def run_naive_fusion(
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
    if time:
        lbls = []
        for i in range(dists.shape[1]):
            lbls.append(
                naive_fusion_isotropic_grid(
                    dists[:, i, ...].transpose(1, 2, 0) - subtract_dist,
                    probs[i, ...],
                    None,
                    prob_thresh,
                    grid=1,
                    max_full_overlaps=max_full_overlaps,
                    no_slicing=no_slicing,
                    show_overlaps=show_overlaps,
                    erase_probs_at_full_overlap=erase_probs_at_full_overlap,
                )
            )
        lbl = np.stack(lbls, axis=0)
    else:
        lbl = naive_fusion_isotropic_grid(
            dists.transpose(1, 2, 0) - subtract_dist,
            probs,
            None,
            prob_thresh,
            grid=1,
            max_full_overlaps=max_full_overlaps,
            no_slicing=no_slicing,
            show_overlaps=show_overlaps,
            erase_probs_at_full_overlap=erase_probs_at_full_overlap,
        )
    return Labels(lbl, name="Merge StarDist Masks")


class MergeStarDistMasksWidget(QWidget):
    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer

        _, w = self.viewer.window.add_plugin_dock_widget(
            "stardist-napari", "StarDist"
        )
        w.prob_thresh.value = 1.0
        w.cnn_output.value = True

        self.viewer.window.add_dock_widget(
            run_naive_fusion,
        )
        # self.setLayout(QHBoxLayout())
        # self.layout().addWidget(run_naive_fusion)
