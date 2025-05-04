import xarray as xr


def alterations_s2(composite: xr.DataArray) -> xr.DataArray:
    """
    Alteration index (Sentinel-2).
    Formula not found in public literature.
    """
    raise NotImplementedError("Alterations S2 formula not found in public sources.")


def fe_silicates_s2(composite: xr.DataArray) -> xr.DataArray:
    """
    Ferrous Silicates index (Sentinel-2): band 11 / band 8A
    Source: Ge et al. (2020) :contentReference[oaicite:0]{index=0}
    """
    return composite.sel(band="B11") / composite.sel(band="B8A")


def fe2o3_s2(composite: xr.DataArray) -> xr.DataArray:
    """
    Ferric iron (Fe₂O₃) index (Sentinel-2): band 11 / band 1
    Source: Ge et al. (2020) :contentReference[oaicite:1]{index=1}
    """
    return composite.sel(band="B11") / composite.sel(band="B01")


def fe3_s2(composite: xr.DataArray) -> xr.DataArray:
    """
    Ferric iron (Fe³⁺) index (Sentinel-2).
    Formula not found in public literature.
    """
    raise NotImplementedError("Fe3+ S2 formula not found in public sources.")


def goethite_s2(composite: xr.DataArray) -> xr.DataArray:
    """
    Goethite index (Sentinel-2): same band 11 / band 1 ratio picks up hematite+goethite mixture
    Source: Ge et al. (2020) :contentReference[oaicite:2]{index=2}
    """
    return composite.sel(band="B11") / composite.sel(band="B01")


# — EMIT-derived mineral indices —
def calcite_emit(composite: xr.DataArray) -> xr.DataArray:
    """
    Calcite index (EMIT).
    Formula not publicly documented.
    """
    raise NotImplementedError("Calcite EMIT index formula not found.")


def clay_minerals_emit(composite: xr.DataArray) -> xr.DataArray:
    """
    Clay Minerals index (EMIT).
    Formula not publicly documented.
    """
    raise NotImplementedError("Clay Minerals EMIT index formula not found.")


def epi_chlo_calc_emit(composite: xr.DataArray) -> xr.DataArray:
    """
    EPI/CHLO/CALC alteration index (EMIT).
    Formula not publicly documented.
    """
    raise NotImplementedError("EPI/CHLO/CALC EMIT index formula not found.")


def feai_emit(composite: xr.DataArray) -> xr.DataArray:
    """
    FEAI index (EMIT).
    Formula not publicly documented.
    """
    raise NotImplementedError("FEAI EMIT index formula not found.")


def fei_emit(composite: xr.DataArray) -> xr.DataArray:
    """
    FEI index (EMIT).
    Formula not publicly documented.
    """
    raise NotImplementedError("FEI EMIT index formula not found.")


def gypsum_emit(composite: xr.DataArray) -> xr.DataArray:
    """
    Gypsum index (EMIT).
    Formula not publicly documented.
    """
    raise NotImplementedError("Gypsum EMIT index formula not found.")


def illite_emit(composite: xr.DataArray) -> xr.DataArray:
    """
    Illite index (EMIT).
    Formula not publicly documented.
    """
    raise NotImplementedError("Illite EMIT index formula not found.")


def iron_oxides_emit(composite: xr.DataArray) -> xr.DataArray:
    """
    Iron Oxides index (EMIT).
    Formula not publicly documented.
    """
    raise NotImplementedError("Iron Oxides EMIT index formula not found.")


def kaolinite1_emit(composite: xr.DataArray) -> xr.DataArray:
    """
    Kaolinite 1 index (EMIT).
    Formula not publicly documented.
    """
    raise NotImplementedError("Kaolinite 1 EMIT index formula not found.")


def kaolinite3_emit(composite: xr.DataArray) -> xr.DataArray:
    """
    Kaolinite 3 index (EMIT).
    Formula not publicly documented.
    """
    raise NotImplementedError("Kaolinite 3 EMIT index formula not found.")


def montmorillonite_emit(composite: xr.DataArray) -> xr.DataArray:
    """
    Montmorillonite index (EMIT).
    Formula not publicly documented.
    """
    raise NotImplementedError("Montmorillonite EMIT index formula not found.")
