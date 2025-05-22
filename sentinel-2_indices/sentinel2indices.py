import xarray as xr

def hematite_goethite_s2(composite: xr.DataArray) -> xr.DataArray:
    """
    Hematite + Goethite index (Sentinel-2): band 6 / band 1
    Source: Ge et al. (2020)
    """
    return composite.sel(band="B06") / composite.sel(band="B01")

def hematite_jarosite_s2(composite: xr.DataArray) -> xr.DataArray:
    """
    Hematite + Jarosite index (Sentinel-2): band 6 / band 8A
    Source: Ge et al. (2020)
    """
    return composite.sel(band="B06") / composite.sel(band="B8A")

def mixed_iron_s2(composite: xr.DataArray) -> xr.DataArray:
    """
    Mixed Iron-bearing Minerals index (Sentinel-2): (band 6 + band 7) / band 8A
    Source: Ge et al. (2020)
    """
    return (composite.sel(band="B06") + composite.sel(band="B07")) / composite.sel(band="B8A")

def ndvi_s2(composite: xr.DataArray) -> xr.DataArray:
    """
    Normalized Difference Vegetation Index (Sentinel-2): (band 8 - band 4) / (band 8 + band 4)
    Source: Rouse et al. (1973)
    """
    nir = composite.sel(band="B08")
    red = composite.sel(band="B04")
    return (nir - red) / (nir + red)