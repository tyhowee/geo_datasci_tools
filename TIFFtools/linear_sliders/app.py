import os
import io
import base64
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import (
    calculate_default_transform,
    reproject,
    Resampling as WarpResampling,
)
import matplotlib.pyplot as plt
import contextily as ctx
import streamlit as st

# --- 1. Define paths to rasters ---
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
    developed_mask_path: "Max number of buildings in each cluster",
    protected_areas_path: "Min distance to nearest protected area (m)",
}

st.set_page_config(
    page_title="Interactive ML & Stripping Ratio Map",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --------------------------------------------------------------------
# 2. Load & cache heavy GIS work once per session
# --------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_data_and_reproject():
    # Reference for grid & metadata
    with rasterio.open(ML_path) as ref:
        ref_meta = ref.meta.copy()
        H, W = ref.height, ref.width
        src_crs = ref.crs
        src_tf = ref.transform

    # Read & resample arrays to reference grid
    arrays = []
    for p in paths:
        with rasterio.open(p) as src:
            resamp = (
                Resampling.nearest if p == developed_mask_path else Resampling.bilinear
            )
            arr = src.read(1, out_shape=(H, W), resampling=resamp).astype(float)
        arrays.append(arr)
    with rasterio.open(stripping_ratio_path) as src:
        strip_native = src.read(
            1, out_shape=(H, W), resampling=Resampling.bilinear
        ).astype(float)

    # Compute Web-Mercator transform & shape
    xmin = ref_meta["transform"][2]
    ymax = ref_meta["transform"][5]
    xmax = xmin + ref_meta["transform"][0] * W
    ymin = ymax + ref_meta["transform"][4] * H
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

    # Reproject maps
    ml_map = reproject_arr(arrays[0], WarpResampling.bilinear)
    infra_maps = [
        reproject_arr(
            arr,
            (
                WarpResampling.nearest
                if p == developed_mask_path
                else WarpResampling.bilinear
            ),
        )
        for arr, p in zip(arrays[1:], paths[1:])
    ]
    stripping_map = reproject_arr(strip_native, WarpResampling.bilinear)

    # Extent in Web-Mercator
    x_min = transform_3857.c
    y_max = transform_3857.f
    x_max = x_min + transform_3857.a * w_3857
    y_min = y_max + transform_3857.e * h_3857
    extent = (x_min, x_max, y_min, y_max)

    # Slider info
    infra_names = [os.path.splitext(os.path.basename(p))[0] for p in paths[1:]]
    infra_mins = [float(np.nanmin(m)) for m in infra_maps]
    infra_maxs = [float(np.nanmax(m)) for m in infra_maps]
    name_to_label = {
        os.path.splitext(os.path.basename(p))[0]: lbl for p, lbl in labels.items()
    }
    protected_name = os.path.splitext(os.path.basename(protected_areas_path))[0]
    developed_name = os.path.splitext(os.path.basename(developed_mask_path))[0]

    return (
        ml_map,
        infra_maps,
        stripping_map,
        extent,
        infra_names,
        infra_mins,
        infra_maxs,
        name_to_label,
        protected_name,
        developed_name,
    )


@st.cache_resource(show_spinner=False)
def load_basemap(extent, provider, zoom=9):
    xmin, xmax, ymin, ymax = extent
    img, bbox = ctx.bounds2img(xmin, ymin, xmax, ymax, zoom=zoom, source=provider)
    return img, bbox


# Load data & basemap
(
    ml_map,
    infra_maps,
    stripping_map,
    extent,
    infra_names,
    infra_mins,
    infra_maxs,
    name_to_label,
    protected_name,
    developed_name,
) = load_data_and_reproject()
try:
    provider = ctx.providers.CartoDB.Positron
except AttributeError:
    provider = ctx.providers.Stamen.Terrain
basemap_img, basemap_bbox = load_basemap(extent, provider)

# --------------------------------------------------------------------
# 3. Downsample for fast plotting
# --------------------------------------------------------------------
H_disp, W_disp = basemap_img.shape[:2]


def downsample(arr, out_h, out_w):
    y_factor = max(1, int(np.floor(arr.shape[0] / out_h)))
    x_factor = max(1, int(np.floor(arr.shape[1] / out_w)))
    return arr[::y_factor, ::x_factor]


ml_preview = downsample(ml_map, H_disp, W_disp)
strip_preview = downsample(stripping_map, H_disp, W_disp)
infra_previews = [downsample(m, H_disp, W_disp) for m in infra_maps]

# --------------------------------------------------------------------
# 4. Sidebar sliders
# --------------------------------------------------------------------
st.sidebar.title("Infrastructure thresholds")
thresholds = {}
for name, mn, mx in zip(infra_names, infra_mins, infra_maxs):
    lbl = name_to_label.get(name, name)
    if name == developed_name:
        val = st.sidebar.slider(lbl, int(mn), int(mx), int(mx), step=1)
    else:
        val = st.sidebar.slider(lbl, float(mn), float(mx), float(mn))
    thresholds[name] = val

thresholds_tuple = tuple(thresholds[name] for name in infra_names)


# --------------------------------------------------------------------
# 5. Cached figure creation
# --------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def make_figure(thresholds_tuple):
    thr_vals = thresholds_tuple
    filtered = ml_preview.copy()
    for name, base_preview, thr in zip(infra_names, infra_previews, thr_vals):
        if name == protected_name:
            filtered[base_preview < thr] = 0
        else:
            filtered[base_preview > thr] = 0
    ml_masked = np.ma.masked_equal(filtered, 0)
    ml_bool = filtered > 0
    strip_masked = np.ma.masked_where(~ml_bool, strip_preview)

    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(20, 10), dpi=100)
    for ax in (ax1, ax2):
        xmin, xmax, ymin, ymax = basemap_bbox
        ax.imshow(basemap_img, extent=basemap_bbox, origin="upper")
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.axis("off")

    im1 = ax1.imshow(ml_masked, extent=extent, origin="upper", cmap="viridis")
    ax1.set_title("ML probability (masked)", fontsize=16, fontweight="bold", pad=8)
    fig.colorbar(im1, label="Probability", ax=ax1, fraction=0.046, pad=0.04)

    im2 = ax2.imshow(strip_masked, extent=extent, origin="upper", cmap="magma")
    ax2.set_title("Stripping ratio (ML-masked)", fontsize=16, fontweight="bold", pad=8)
    fig.colorbar(im2, label="Stripping Ratio", ax=ax2, fraction=0.046, pad=0.04)

    plt.tight_layout()
    return fig


# --------------------------------------------------------------------
# 6. Display (embed high-res PNG)
# --------------------------------------------------------------------
st.title("Interactive ML & Stripping Ratio Map")
fig = make_figure(thresholds_tuple)
# Save to high-res PNG in memory and embed via HTML for crisp display
buf = io.BytesIO()
fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
buf.seek(0)
img_b64 = base64.b64encode(buf.read()).decode()
st.markdown(
    f"<img style='max-width: 100%; height: auto;' src='data:image/png;base64,{img_b64}'/>",
    unsafe_allow_html=True,
)
