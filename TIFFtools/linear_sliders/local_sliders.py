"""
Script to load rasters and display interactive slider plot for linear combination filtering.
Usage: python linear_combination_sliders_slim.py
"""

import os
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import (
    calculate_default_transform,
    reproject,
    Resampling as WarpResampling,
)
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# 1. Hardcoded file paths
ML_path = r"C:\Users\TyHow\MinersAI Dropbox\Tyler Howe\Sibelco_Stuff\linear_combination_layers\averaged_probability_map_smoothed_thresholded_95th_percentile.tif"
roads_path = r"C:\Users\TyHow\MinersAI Dropbox\Tyler Howe\Sibelco_Stuff\linear_combination_layers\infrastructure\major_roads.tif"
power_path = r"C:\Users\TyHow\MinersAI Dropbox\Tyler Howe\Sibelco_Stuff\linear_combination_layers\infrastructure\powerlines.tif"
rail_path = r"C:\Users\TyHow\MinersAI Dropbox\Tyler Howe\Sibelco_Stuff\linear_combination_layers\infrastructure\railways.tif"
developed_mask_path = r"C:\Users\TyHow\MinersAI Dropbox\Tyler Howe\Sibelco_Stuff\linear_combination_layers\infrastructure\town_mask_continuous.tif"
protected_areas_path = r"C:\Users\TyHow\MinersAI Dropbox\Tyler Howe\Sibelco_Stuff\linear_combination_layers\infrastructure\protected_areas.tif"
stripping_ratio_path = r"C:\Users\TyHow\MinersAI Dropbox\Tyler Howe\Sibelco_Stuff\linear_combination_layers\stripping_ratio.tif"

paths = [
    ML_path,
    roads_path,
    power_path,
    rail_path,
    developed_mask_path,
    protected_areas_path,
]
labels = {
    ML_path: "Smoothed ML probability map (95th percentile)",
    roads_path: "Max distance to nearest road (m)",
    power_path: "Max distance to nearest power line (m)",
    rail_path: "Max distance to nearest railway (m)",
    developed_mask_path: "Max number of buildings allowed in each developed cluster",
    protected_areas_path: "Min distance to nearest protected area (m)",
}


def main():
    # Load reference raster
    with rasterio.open(ML_path) as ref:
        meta = ref.meta.copy()
        H, W = ref.height, ref.width
        src_crs, src_tf = ref.crs, ref.transform

    # Read and resample all layers
    arrays = []
    for p in paths:
        with rasterio.open(p) as src:
            method = (
                Resampling.nearest if p == developed_mask_path else Resampling.bilinear
            )
            arrays.append(
                src.read(1, out_shape=(H, W), resampling=method).astype(float)
            )
    with rasterio.open(stripping_ratio_path) as src:
        strip_native = src.read(
            1, out_shape=(H, W), resampling=Resampling.bilinear
        ).astype(float)

    # Setup web-mercator transform
    xmin = meta["transform"][2]
    ymax = meta["transform"][5]
    xmax = xmin + meta["transform"][0] * W
    ymin = ymax + meta["transform"][4] * H
    dst_crs = "EPSG:3857"
    transform_3857, w_3857, h_3857 = calculate_default_transform(
        src_crs, dst_crs, W, H, left=xmin, bottom=ymin, right=xmax, top=ymax
    )

    def reproject_arr(arr, method):
        dst = np.zeros((h_3857, w_3857), dtype=arr.dtype)
        reproject(
            source=arr,
            destination=dst,
            src_transform=src_tf,
            src_crs=src_crs,
            dst_transform=transform_3857,
            dst_crs=dst_crs,
            resampling=method,
        )
        return dst

    # Reproject rasters
    ml_map = reproject_arr(arrays[0], WarpResampling.bilinear)
    infra_maps = [
        reproject_arr(
            a,
            (
                WarpResampling.nearest
                if p == developed_mask_path
                else WarpResampling.bilinear
            ),
        )
        for a, p in zip(arrays[1:], paths[1:])
    ]
    strip_map = reproject_arr(strip_native, WarpResampling.bilinear)

    # Slider metadata
    infra_names = [os.path.splitext(os.path.basename(p))[0] for p in paths[1:]]
    infra_mins = [float(np.nanmin(m)) for m in infra_maps]
    infra_maxs = [float(np.nanmax(m)) for m in infra_maps]
    n = len(infra_names)

    # Compute extent
    left, top = transform_3857[2], transform_3857[5]
    right = left + transform_3857[0] * w_3857
    bottom = top + transform_3857[4] * h_3857
    extent = (left, right, bottom, top)

    # Layout with GridSpec + constrained_layout
    plt.rcParams["figure.constrained_layout.use"] = True
    fig = plt.figure(figsize=(12, 8), constrained_layout=True)

    # — Here’s the only change: give the sliders row a small, fixed height (1 unit)
    # instead of "n" units. You can make that second number even smaller (e.g. 0.5)
    # if you want them slimmer yet.
    gs0 = fig.add_gridspec(2, 1, height_ratios=[3, 0.5])

    # Maps sub-grid
    gs_maps = gs0[0].subgridspec(1, 2)
    ax1 = fig.add_subplot(gs_maps[0, 0])
    ax2 = fig.add_subplot(gs_maps[0, 1])
    # Slider sub-grid
    gs_sliders = gs0[1].subgridspec(n, 1)
    slider_axes = [fig.add_subplot(gs_sliders[i, 0]) for i in range(n)]

    # Plot ML map
    ml_masked = np.ma.masked_equal(ml_map, 0)
    im1 = ax1.imshow(ml_masked, extent=extent, origin="upper", cmap="viridis")
    ax1.set_title("ML probability (masked within thresholds)")
    fig.colorbar(im1, ax=ax1, label="Probability")

    # Plot stripping map
    strip_masked = np.ma.masked_where(~(ml_map > 0), strip_map)
    im2 = ax2.imshow(strip_masked, extent=extent, origin="upper", cmap="magma")
    ax2.set_title("Stripping ratio (ML-masked)")
    fig.colorbar(im2, ax=ax2, label="Stripping Ratio")

    # Create sliders
    sliders = []
    for ax, p, mn, mx in zip(slider_axes, paths[1:], infra_mins, infra_maxs):
        default = mn if p == protected_areas_path else mx
        s = Slider(ax, labels[p], mn, mx, valinit=default)
        sliders.append((p, s))

    # Update callback
    def update(val):
        filt = ml_map.copy()
        for p, s in sliders:
            thr = s.val
            layer = infra_maps[paths[1:].index(p)]
            if p == protected_areas_path:
                filt[layer < thr] = 0
            else:
                filt[layer > thr] = 0
        im1.set_data(np.ma.masked_equal(filt, 0))
        im2.set_data(np.ma.masked_where(~(filt > 0), strip_map))
        fig.canvas.draw_idle()

    for _, s in sliders:
        s.on_changed(update)

    plt.show()


if __name__ == "__main__":
    main()
