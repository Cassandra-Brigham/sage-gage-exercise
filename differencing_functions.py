from __future__ import annotations

# Standard library
import json
import os
import re
import tempfile
import subprocess
import time
import uuid
import warnings
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

# Third-party
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import requests
import rasterio
import rioxarray as rio
from rasterio.warp import Resampling
from osgeo import gdal
import geopandas as gpd
from shapely.geometry import Polygon, LineString, MultiPolygon, shape, box, mapping
from shapely.ops import unary_union, transform
from scipy import stats
from scipy.stats import norm
from scipy.optimize import curve_fit
from numba import njit, prange
from pyproj import Transformer, CRS
from pyproj.aoi import AreaOfInterest
from pyproj.database import query_utm_crs_info
from ipyleaflet import Map, GeomanDrawControl, GeoJSON, LegendControl, basemaps, ScaleControl
import colormaps as cmaps
from adjustText import adjust_text
from sklearn.utils import resample
from esda.moran import Moran
from libpysal.weights import KNN
from splot.esda import moran_scatterplot

import numpy as np
import rasterio
import rioxarray as rio
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

# Optional PDAL
try:
    import pdal  # noqa: F401
    _PDAL_AVAILABLE = True
except ImportError:
    _PDAL_AVAILABLE = False

# GDAL config
gdal.UseExceptions()

# ----------------------------------------------------------------------
# Raster helpers
# ----------------------------------------------------------------------

@dataclass
class Raster:
    """Lazy wrapper around a raster file with convenience plotting."""

    path: Path | str
    _data: rio.DataArray | None = None

    def __post_init__(self):
        self.path = Path(self.path)
        if not self.path.exists():
            raise FileNotFoundError(self.path)

    # ---- data access -------------------------------------------------
    @property
    def data(self):
        if self._data is None:
            self._data = rio.open_rasterio(self.path, masked=True)
        return self._data

    @property
    def shape(self) -> Tuple[int, int, int]:
        return self.data.shape

    @property
    def crs(self):
        return self.data.rio.crs

    # -------------------------- utilities ----------------------------
    def reproject_to(self, target: "Raster") -> "Raster":
        out = self.path.with_name(f"{self.path.stem}_reproj_{uuid.uuid4().hex[:6]}.tif")
        self.data.rio.reproject_match(target.data).rio.to_raster(out)
        return Raster(out)

    # --------------------------- plotting ----------------------------
    def plot(self, *, ax=None, cmap="viridis", vmin=None, vmax=None, title=None, **imshow_kw):
        if ax is None:
            _, ax = plt.subplots()
        arr = self.data.squeeze().values
        im = ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax, **imshow_kw)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_xticks([]); ax.set_yticks([])
        if title:
            ax.set_title(title)
        return ax
    
@dataclass
class RasterPair:
    """Pair two rasters, ensure alignment, and provide plotting."""

    raster1: Raster | str
    raster2: Raster | str
    _aligned: bool = False

    def __post_init__(self):
        self.raster1 = Raster(self.raster1) if not isinstance(self.raster1, Raster) else self.raster1
        self.raster2 = Raster(self.raster2) if not isinstance(self.raster2, Raster) else self.raster2

    # ---------------------- alignment checker ------------------------
    def _align(self):
        if self._aligned:
            return
        def key(r: Raster):
            return (r.crs, r.data.rio.transform(), r.shape)
        if key(self.raster1) != key(self.raster2):
            larger, smaller = ((self.raster1, self.raster2) if np.prod(self.raster1.shape) > np.prod(self.raster2.shape) else (self.raster2, self.raster1))
            warnings.warn("Raster grid mismatch – reprojecting the larger raster to match the smaller.")
            reproj = larger.reproject_to(smaller)
            if larger is self.raster1:
                self.raster1 = reproj
            else:
                self.raster2 = reproj
        self._aligned = True

    raster1_data = property(lambda self: (self._align(), self.raster1.data)[1])
    raster2_data = property(lambda self: (self._align(), self.raster2.data)[1])

       # --------------------------- plotting ----------------------------
    def plot_pair(
        self,
        *,
        overlay: Raster | None = None,
        overlay_alpha: float = 0.5,
        vmin=None, 
        vmax=None,
        titles=("Raster 1", "Raster 2"),
        base_cmap="viridis",
        overlay_cmap="magma",
        legend: str | None = None,
    ) -> plt.Figure:
        """Plot two rasters side-by-side; optionally overlay a third raster
        and draw a single shared colorbar legend."""
        self._align()

        fig, axs = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

        # --- Plot base rasters manually, capture the first image for legend ---
        arr1 = self.raster1.data.squeeze().values
        arr2 = self.raster2.data.squeeze().values

        im = axs[0].imshow(arr1, cmap=base_cmap, vmin=vmin, vmax=vmax)
        axs[0].set_title(titles[0])
        axs[0].axis("off")

        axs[1].imshow(arr2, cmap=base_cmap, vmin=im.norm.vmin, vmax=im.norm.vmax)
        axs[1].set_title(titles[1])
        axs[1].axis("off")

        # --- Overlay, if present ---
        if overlay is not None:
            for base, ax in zip((self.raster1, self.raster2), axs):
                ov = overlay
                if ov.crs != base.crs or ov.shape[1:] != base.shape[1:]:
                    ov = ov.reproject_to(base)
                ax.imshow(ov.data.squeeze().values,
                          cmap=overlay_cmap,
                          alpha=overlay_alpha,
                          vmin=None, vmax=None)

        # --- Single shared colorbar ---
        if legend:
            cbar = fig.colorbar(im, ax=axs, fraction=0.046, pad=0.04)
            cbar.set_label(legend)

        return fig

# ----------------------------------------------------------------------
# DataAccess
# ----------------------------------------------------------------------
class DataAccess:
    """Wrapper for interactive or programmatic AOI definition."""

    # --------------------- static helpers ------------------------------
    @staticmethod
    def _coords_to_wkt(coords):
        return ", ".join(
            ", ".join(f"{x}, {y}" for x, y in ring) for ring in coords
        )

    @classmethod
    def geojson_to_OTwkt(cls, gj):
        if gj["type"] != "Polygon":
            raise ValueError("Input must be a Polygon GeoJSON")
        return cls._coords_to_wkt(gj["coordinates"])

    # --------------------- AOI via map draw ----------------------------
    def init_ot_catalog_map(
        self,
        center=(39.8283, -98.5795),
        zoom=3,
        layers=(
            ("3DEP", "https://raw.githubusercontent.com/OpenTopography/Data_Catalog_Spatial_Boundaries/main/usgs_3dep_boundaries.geojson", "#228B22"),
            ("NOAA", "https://raw.githubusercontent.com/OpenTopography/Data_Catalog_Spatial_Boundaries/main/noaa_coastal_lidar_boundaries.geojson", "#0000CD"),
            ("OpenTopography", "https://raw.githubusercontent.com/OpenTopography/Data_Catalog_Spatial_Boundaries/main/OT_PC_boundaries.geojson", "#fca45d"),
        ),
    ):
        """Return ipyleaflet Map with draw control and catalog layers."""
        self.bounds: dict[str, Any] = {}
        self.polygon: dict[str, Any] = {}

        def _on_draw(control, action, geo_json):
            feats = geo_json if isinstance(geo_json, list) else [geo_json]
            shapes = [shape(f["geometry"]) for f in feats]
            wkt_list = [self.geojson_to_OTwkt(f["geometry"]) for f in feats]
            merged = unary_union(shapes)
            minx, miny, maxx, maxy = merged.bounds
            self.bounds.update(dict(south=miny, west=minx, north=maxy, east=maxx, polygon_wkt=wkt_list))
            self.polygon.update(dict(merged_polygon=merged, all_polys=shapes))
            print("AOI bounds:", self.bounds)

        m = Map(center=center, zoom=zoom, basemap=basemaps.Esri.WorldTopoMap)
        dc = GeomanDrawControl(rectangle={"pathOptions": {"color": "#fca45d", "fillColor": "#fca45d"}},
                               polygon={"pathOptions": {"color": "#6be5c3", "fillColor": "#6be5c3"}})
        dc.polyline = dc.circlemarker = {}
        dc.on_draw(_on_draw)
        m.add_control(dc)

        for name, url, color in layers:
            gj = GeoJSON(data=requests.get(url).json(), name=name, style={"color": color})
            m.add_layer(gj)
        m.add_control(LegendControl({n: c for n, _, c in layers}, name="Legend"))
        m.add_control(ScaleControl(position="bottomleft", metric=True, imperial=False))
        return m

    # --------------------- AOI via manual bounds -----------------------
    def define_bounds_manual(self, south, north, west, east):
        poly = box(west, south, east, north)
        self.bounds = dict(south=south, north=north, west=west, east=east,
                           polygon_wkt=[self.geojson_to_OTwkt(mapping(poly))])
        self.polygon = dict(merged_polygon=poly, all_polys=[poly])
        return self.bounds

    # --------------------- AOI via uploaded vector ---------------------
    def define_bounds_from_file(self, vector_path: str, target_crs="EPSG:4326"):
        gdf = gpd.read_file(vector_path)
        if gdf.empty:
            raise ValueError("No geometries in file")
        if gdf.crs is None:
            raise ValueError("Input CRS undefined")
        if gdf.crs.to_string() != target_crs:
            gdf = gdf.to_crs(target_crs)
        merged = unary_union(gdf.geometry)
        minx, miny, maxx, maxy = merged.bounds
        wkts = [self.geojson_to_OTwkt(mapping(geom)) for geom in gdf.geometry]
        self.bounds = dict(south=miny, west=minx, north=maxy, east=maxx, polygon_wkt=wkts)
        self.polygon = dict(merged_polygon=merged, all_polys=list(gdf.geometry))
        return self.bounds

# ----------------------------------------------------------------------
# OpenTopographyQuery
# ----------------------------------------------------------------------
class OpenTopographyQuery:
    def __init__(self, data_access: DataAccess):
        self.da = data_access
        self.catalog_df: pd.DataFrame | None = None

    @staticmethod
    def _clean(name: str) -> str:
        return re.sub(r"[^\w]+", "_", name)

    def query_catalog(
        self,
        product_format="PointCloud",
        include_federated=True,
        detail=False,
        save_as="results.json",
        url="https://portal.opentopography.org/API/otCatalog",
    ) -> pd.DataFrame:
        if not getattr(self.da, "bounds", None):
            raise ValueError("AOI not defined")
        params = dict(
            productFormat=product_format,
            detail=str(detail).lower(),
            outputFormat="json",
            include_federated=str(include_federated).lower(),
        )
        if self.da.bounds.get("polygon_wkt"):
            params["polygon"] = self.da.bounds["polygon_wkt"]
        else:
            params.update(dict(minx=self.da.bounds["west"], miny=self.da.bounds["south"],
                               maxx=self.da.bounds["east"], maxy=self.da.bounds["north"]))
        r = requests.get(url, params=params, timeout=60)
        r.raise_for_status()
        if save_as:
            Path(save_as).write_bytes(r.content)
            
        data = r.json()
        # ---- flatten into DataFrame (unchanged) ----
        rows = []
        for ds in data["Datasets"]:
            meta = ds["Dataset"]
            rows.append(
                {
                    "Name":           meta["name"],
                    "ID type":        meta["identifier"]["propertyID"],
                    "Data Source":    "usgs" if "USGS" in meta["identifier"]["propertyID"] or "usgs" in meta["identifier"]["propertyID"] else
                                      "noaa" if "NOAA" in meta["identifier"]["propertyID"] or "noaa" in meta["identifier"]["propertyID"] else "ot",
                    "Property ID":    meta["identifier"]["value"],
                    "Horizontal EPSG":
                        next((p["value"] for p in meta["spatialCoverage"]["additionalProperty"]
                            if p["name"] == "EPSG (Horizontal)"), None),
                    "Vertical Coordinates":
                        next((p["value"] for p in meta["spatialCoverage"]["additionalProperty"]
                            if p["name"] == "Vertical Coordinates"), None),
                    "Clean Name":     self._clean(meta["name"]),
                }
            )
        

        self.catalog_df =  pd.DataFrame(rows)
        
        return self.catalog_df
        

    # shorthand to select compare / reference rows by DataFrame index
    def pick(self, idx_compare: int, idx_reference: int):
        df = self.catalog_df
        self.compare = df.iloc[idx_compare]
        self.reference = df.iloc[idx_reference]
        self.compare_name = self.catalog_df["Name"].iloc[idx_compare]
        self.compare_data_source = self.catalog_df["Data Source"].iloc[idx_compare]
        self.compare_property_id = self.catalog_df["Property ID"].iloc[idx_compare]
        self.compare_horizontal_crs = self.catalog_df["Horizontal EPSG"].iloc[idx_compare]
        self.compare_vertical_crs = self.catalog_df["Vertical Coordinates"].iloc[idx_compare]
        self.compare_clean_name = self.catalog_df["Clean Name"].iloc[idx_compare]
        self.reference_name = self.catalog_df["Name"].iloc[idx_reference]
        self.reference_data_source = self.catalog_df["Data Source"].iloc[idx_reference]
        self.reference_property_id = self.catalog_df["Property ID"].iloc[idx_reference]
        self.reference_horizontal_crs = self.catalog_df["Horizontal EPSG"].iloc[idx_reference]
        self.reference_vertical_crs = self.catalog_df["Vertical Coordinates"].iloc[idx_reference]
        self.reference_clean_name = self.catalog_df["Clean Name"].iloc[idx_reference]
        if self.compare["Vertical Coordinates"] != self.reference["Vertical Coordinates"]:
            print("⚠️  Vertical CRSs differ between datasets")
        return self.compare, self.reference

# ------------------------------------------------------------------------------------------
# Download data and make DEMs
# ------------------------------------------------------------------------------------------
class GetDEMs:
    """Generate DEMs from local LAZ/point files or AWS EPT sources."""

    def __init__(self, data_access, ot_query):
        self.da = data_access
        self.ot = ot_query

    # ------------------------------------------------------------------
    #                  Raster gap‑fill utility                           
    # ------------------------------------------------------------------
    @staticmethod
    def fill_no_data(input_file, output_file, *, method="idw", nodata=-9999, max_dist=100, smooth_iter=0):
        ds = gdal.Open(input_file, gdal.GA_ReadOnly)
        band = ds.GetRasterBand(1)
        arr = band.ReadAsArray()
        mask = arr == nodata

        def _interpolate(other):
            valid = np.where(~mask)
            nod = np.where(mask)
            coords = np.column_stack(valid)
            vals = arr[valid]
            if other == "nearest":
                return griddata(coords, vals, np.column_stack(nod), method="nearest")
            if other == "linear":
                return griddata(coords, vals, np.column_stack(nod), method="linear")
            if other == "cubic":
                return griddata(coords, vals, np.column_stack(nod), method="cubic")
            if other == "spline":
                rbf = Rbf(coords[:, 0], coords[:, 1], vals, function="thin_plate")
                return rbf(nod[0], nod[1])
            raise ValueError("Unknown method")

        if method == "idw":
            mem = gdal.GetDriverByName("MEM").CreateCopy("", ds, 0)
            gdal.FillNodata(mem.GetRasterBand(1), None, max_dist, smooth_iter)
            filled = mem.GetRasterBand(1).ReadAsArray()
        else:
            filled = arr.copy()
            filled_vals = _interpolate(method)
            filled[np.where(mask)] = filled_vals

        drv = gdal.GetDriverByName("GTiff")
        out = drv.Create(output_file, ds.RasterXSize, ds.RasterYSize, 1, gdal.GDT_Float32)
        out.SetGeoTransform(ds.GetGeoTransform())
        out.SetProjection(ds.GetProjection())
        out.GetRasterBand(1).WriteArray(filled)
        out.GetRasterBand(1).SetNoDataValue(nodata)
        out.FlushCache()

    # ------------------------------------------------------------------
    #                     Internal PDAL helpers                         
    # ------------------------------------------------------------------
    @staticmethod
    def _writer_gdal(filename, *, grid_method="idw", res=1.0, driver="GTiff"):
        return {
            "type": "writers.gdal",
            "filename": filename,
            "gdaldriver": driver,
            "nodata": -9999,
            "output_type": grid_method,
            "resolution": float(res),
            "radius": 2 * float(res),
            "gdalopts": "COMPRESS=LZW,TILED=YES,blockxsize=256,blockysize=256,COPY_SRC_OVERVIEWS=YES",
        }

    @staticmethod
    def _writer_las(name, ext):
        if ext not in {"las", "laz"}:
            raise ValueError("pc_outType must be 'las' or 'laz'")
        w = {"type": "writers.las", "filename": f"{name}.{ext}"}
        if ext == "laz":
            w["compression"] = "laszip"
        return w

    # ----------------------- Local file pipelines ---------------------
    @staticmethod
    def build_pdal_pipeline_from_file(filename, extent, filterNoise=False, reclassify=False, savePointCloud=True, outCRS=3857, 
                            pc_outName='filter_test', pc_outType='laz'):
        # Initialize the pipeline with reading and cropping stages
        pointcloud_pipeline = [
            {
                "type": "readers.las",
                "filename": filename
            },
            {
                "type": "filters.crop",
                "polygon": extent.wkt
            }
        ]
        
        # Optionally add a noise filter stage
        if filterNoise:
            pointcloud_pipeline.append({
                "type": "filters.range",
                "limits": "Classification![7:7], Classification![18:18]"
            })
        
        # Optionally add reclassification stages
        if reclassify:
            pointcloud_pipeline += [
                {"type": "filters.assign", "value": "Classification = 0"},
                {"type": "filters.smrf"},
                {"type": "filters.range", "limits": "Classification[2:2]"}
            ]
        
        # Add reprojection stage
        pointcloud_pipeline.append({
            "type": "filters.reprojection",
            "out_srs": f"EPSG:{outCRS}"
        })
        
        # Optionally add a save point cloud stage
        if savePointCloud:
            if pc_outType not in ['las', 'laz']:
                raise Exception("pc_outType must be 'las' or 'laz'.")
            
            writer_stage = {
                "type": "writers.las",
                "filename": f"{pc_outName}.{pc_outType}"
            }
            if pc_outType == 'laz':
                writer_stage["compression"] = "laszip"
            
            pointcloud_pipeline.append(writer_stage)
            
        return pointcloud_pipeline
    
    def make_DEM_pipeline_from_file(self, filename, extent, dem_resolution,
                        filterNoise=True, reclassify=False, savePointCloud=False, outCRS=3857,
                        pc_outName='filter_test', pc_outType='laz', demType='dtm', gridMethod='idw', 
                        dem_outName='dem_test', dem_outExt='tif', driver="GTiff"):
        # Build the base point cloud pipeline using the provided parameters
        pointcloud_pipeline = self.build_pdal_pipeline_from_file(filename, extent, filterNoise, reclassify, savePointCloud, outCRS, pc_outName, pc_outType)
        
        # Prepare the base pipeline dictionary
        dem_pipeline = {
            "pipeline": pointcloud_pipeline
        }

        # Add appropriate stages based on DEM type
        if demType == 'dsm':
            # Directly add the DSM writer stage
            dem_pipeline['pipeline'].append({
                "type": "writers.gdal",
                "filename": f"{dem_outName}.{dem_outExt}",
                "gdaldriver": driver,
                "nodata": -9999,
                "output_type": gridMethod,
                "resolution": float(dem_resolution),
                "radius": 2*float(dem_resolution),
                "gdalopts": "COMPRESS=LZW,TILED=YES,blockxsize=256,blockysize=256,COPY_SRC_OVERVIEWS=YES"
            })
        
        elif demType == 'dtm':
            # Add a filter to keep only ground points
            dem_pipeline['pipeline'].append({
                "type": "filters.range",
                "limits": "Classification[2:2]"
            })

            # Add the DTM writer stage
            dem_pipeline['pipeline'].append({
                "type": "writers.gdal",
                "filename": f"{dem_outName}.{dem_outExt}",
                "gdaldriver": driver,
                "nodata": -9999,
                "output_type": gridMethod,
                "resolution": float(dem_resolution),
                "radius": 2*float(dem_resolution),
                "gdalopts": "COMPRESS=LZW,TILED=YES,blockxsize=256,blockysize=256,COPY_SRC_OVERVIEWS=YES"
            })
        else:
            raise Exception("demType must be 'dsm' or 'dtm'.")
        
        return dem_pipeline

    # ----------------------- AWS EPT pipeline helpers -----------------
    @staticmethod
    def build_aws_pdal_pipeline(extent_epsg3857, property_ids, pc_resolution, data_source, filterNoise = False,
                            reclassify = False, savePointCloud = True, outCRS = 3857, pc_outName = 'filter_test', 
                            pc_outType = 'laz'):
        readers = []
        for id in property_ids:
            if data_source == 'usgs':
                url = f"https://s3-us-west-2.amazonaws.com/usgs-lidar-public/{id}/ept.json"
            elif data_source == 'noaa':
                stac_url = f"https://noaa-nos-coastal-lidar-pds.s3.amazonaws.com/entwine/stac/DigitalCoast_mission_{id}.json"
                response = requests.get(stac_url)
                data = response.json()
                url = data['assets']['ept']['href']
            else:
                raise ValueError("Invalid dataset source. Must be 'usgs' or 'noaa'.")

            reader = {
                "type": "readers.ept",
                "filename": str(url),
                "polygon": str(extent_epsg3857),
                "requests": 3,
                "resolution": pc_resolution
            }
            readers.append(reader)
            
        pointcloud_pipeline = {
                "pipeline":
                    readers
        }
        
        if filterNoise == True:
            
            filter_stage = {
                "type":"filters.range",
                "limits":"Classification![7:7], Classification![18:18]"
            }
            
            pointcloud_pipeline['pipeline'].append(filter_stage)
        
        if reclassify == True:
            
            remove_classes_stage = {
                "type":"filters.assign",
                "value":"Classification = 0"
            }
            
            classify_ground_stage = {
                "type":"filters.smrf"
            }
            
            reclass_stage = {
                "type":"filters.range",
                "limits":"Classification[2:2]"
            }

        
            pointcloud_pipeline['pipeline'].append(remove_classes_stage)
            pointcloud_pipeline['pipeline'].append(classify_ground_stage)
            pointcloud_pipeline['pipeline'].append(reclass_stage)
            
        reprojection_stage = {
            "type":"filters.reprojection",
            "out_srs":"EPSG:{}".format(outCRS)
        }
        
        pointcloud_pipeline['pipeline'].append(reprojection_stage)
        
        if savePointCloud == True:
            
            if pc_outType == 'las':
                savePC_stage = {
                    "type": "writers.las",
                    "filename": str(pc_outName)+'.'+ str(pc_outType),
                }
            elif pc_outType == 'laz':    
                savePC_stage = {
                    "type": "writers.las",
                    "compression": "laszip",
                    "filename": str(pc_outName)+'.'+ str(pc_outType),
                }
            else:
                raise Exception("pc_outType must be 'las' or 'laz'.")

            pointcloud_pipeline['pipeline'].append(savePC_stage)
            
        return pointcloud_pipeline
    
    def make_DEM_pipeline_aws(self, extent_epsg3857, property_ids, pc_resolution, dem_resolution, data_source = "usgs",
                        filterNoise = True, reclassify = True, savePointCloud = False, outCRS = 3857,
                        pc_outName = 'filter_test', pc_outType = 'laz', demType = 'dtm', gridMethod = 'idw', 
                        dem_outName = 'dem_test', dem_outExt = 'tif', driver = "GTiff"):
        
        dem_pipeline = self.build_aws_pdal_pipeline(extent_epsg3857, property_ids, pc_resolution, data_source,
                                                filterNoise, reclassify, savePointCloud, outCRS, pc_outName, pc_outType)
        
        
        if demType == 'dsm':
            dem_stage = {
                    "type":"writers.gdal",
                    "filename":str(dem_outName)+ '.' + str(dem_outExt),
                    "gdaldriver":driver,
                    "nodata":-9999,
                    "output_type":gridMethod,
                    "resolution":float(dem_resolution),
                    "gdalopts":"COMPRESS=LZW,TILED=YES,blockxsize=256,blockysize=256,COPY_SRC_OVERVIEWS=YES"
            }
        
        elif demType == 'dtm':
            groundfilter_stage = {
                    "type":"filters.range",
                    "limits":"Classification[2:2]"
            }

            dem_pipeline['pipeline'].append(groundfilter_stage)

            dem_stage = {
                    "type":"writers.gdal",
                    "filename":str(dem_outName)+ '.' + str(dem_outExt),
                    "gdaldriver":driver,
                    "nodata":-9999,
                    "output_type":gridMethod,
                    "resolution":float(dem_resolution),
                    "gdalopts":"COMPRESS=LZW,TILED=YES,blockxsize=256,blockysize=256,COPY_SRC_OVERVIEWS=YES"
            }
        
        else:
            raise Exception("demType must be 'dsm' or 'dtm'.")
            
            
        dem_pipeline['pipeline'].append(dem_stage)
        
        return dem_pipeline

    # ------------------------------------------------------------------
    #               Helper: native UTM from AOI bounds                   
    # ------------------------------------------------------------------
    @staticmethod
    def native_utm_crs_from_aoi_bounds(bounds,datum):
        """
        Get the native UTM coordinate reference system from the 

        :param bounds: shapely Polygon of bounding box in EPSG:4326 CRS
        :param datum: string with datum name (e.g., "WGS84")
        :return: UTM CRS code
        """
        utm_crs_list = query_utm_crs_info(
            datum_name=datum,
            area_of_interest=AreaOfInterest(
                west_lon_degree=bounds["west"],
                south_lat_degree=bounds["south"],
                east_lon_degree=bounds["east"],
                north_lat_degree=bounds["north"],
            ),
        )
        utm_crs = CRS.from_epsg(utm_crs_list[0].code)
        return utm_crs

    @staticmethod
    def reproject_polygon(polygon, current_epsg, target_epsg):
        """
        Reprojects a Shapely polygon from one EPSG code to another.

        :param polygon: Shapely Polygon object.
        :param current_epsg: EPSG code of the polygon's current coordinate system.
        :param target_epsg: EPSG code of the target coordinate system.
        :return: A new Shapely Polygon object in the target coordinate system.
        """
        # Create a transformer between the current CRS and the target CRS
        transformer = Transformer.from_crs(f"EPSG:{current_epsg}", f"EPSG:{target_epsg}", always_xy=True)
        
        # Function to apply the transformation to each coordinate
        def apply_transform(x, y):
            return transformer.transform(x, y)

        # Apply the transformation to the polygon
        reprojected_polygon = transform(apply_transform, polygon)
        
        return reprojected_polygon

    # ------------------------------------------------------------------
    #          End‑to‑end driver to download data & create DEMs          
    # ------------------------------------------------------------------

    def dem_download_workflow(
        self,
        folder,
        output_name,                    # Desired generic output name for files on user's local file system (w/o extension, modifiers like "_DTM", "_DSM" will be added depending on product created)       
        dem_resolution = 1.0,           # Desired grid size (in meters) for output raster DEM
        dataset_type = "compare",       # Whether dataset is compare or reference dataset    
        filterNoise = True,             # Option to remove points from USGS Class 7 (Low Noise) and Class 18 (High Noise).
        reclassify = False,         
        savePointCloud = False,         
        pc_resolution = 0.1,            # The desired resolution of the pointcloud based on the following definition: 
                                        #        A point resolution limit to select, expressed as a grid cell edge length. 
                                        #        Units correspond to resource coordinate system units. For example, 
                                        #        for a coordinate system expressed in meters, a resolution value of 0.1 
                                        #        will select points up to a ground resolution of 100 points per square meter.
                                        #        The resulting resolution may not be exactly this value: the minimum possible 
                                        #        resolution that is at least as precise as the requested resolution will be selected. 
                                        #        Therefore the result may be a bit more precise than requested. 
                                        # Source: https://pdal.io/stages/readers.ept.html#readers-ept
        outCRS = "WGS84 UTM",           # Output coordinate reference systemt (CRS), specified by ESPG code (e.g., 3857 - Web Mercator)
        method="idw",                   # method for gap-filling
        nodata=-9999,                   # no data values
        max_dist=100,                   # max distance to consider for gap filling
        smooth_iter=0                   # number of smoothing iterations
    ):
    
        self.initial_compare_dataset_crs = int(self.ot.compare_horizontal_crs)
        self.initial_reference_dataset_crs = int(self.ot.reference_horizontal_crs)

        self.target_compare_dataset_crs = self.native_utm_crs_from_aoi_bounds(self.da.bounds,"WGS84").to_epsg()
        self.target_reference_dataset_crs = self.native_utm_crs_from_aoi_bounds(self.da.bounds,"WGS84").to_epsg()

        self.bounds_polygon_epsg_initial_compare_crs = self.reproject_polygon(self.da.polygon["merged_polygon"], 4326, self.initial_compare_dataset_crs)
        self.bounds_polygon_epsg_initial_reference_crs = self.reproject_polygon(self.da.polygon["merged_polygon"], 4326, self.initial_reference_dataset_crs)
        
    
        if dataset_type == "compare":
            bounds_polygon_epsg_initial_crs = self.bounds_polygon_epsg_initial_compare_crs
            data_source_ = self.ot.compare_data_source
            dataset_id = self.ot.compare_property_id
            dataset_crs_ = self.target_compare_dataset_crs
            self.compare_dtm_path = folder+output_name+'_'+dataset_type+'_DTM.tif'
            self.compare_dsm_path = folder+output_name+'_'+dataset_type+'_DSM.tif'
               
        elif dataset_type == "reference":
            data_source_ = self.ot.reference_data_source
            dataset_id = self.ot.reference_property_id
            bounds_polygon_epsg_initial_crs = self.bounds_polygon_epsg_initial_reference_crs
            dataset_crs_ = self.target_reference_dataset_crs
            self.reference_dtm_path = folder+output_name+'_'+dataset_type+'_DTM.tif'
            self.reference_dsm_path = folder+output_name+'_'+dataset_type+'_DSM.tif'
            
        else:
            raise ValueError("dataset_type must be either 'compare' or 'reference'")

        if outCRS == "WGS84 UTM":
            dataset_crs = dataset_crs_
        else:
            dataset_crs = outCRS
        
        if data_source_ == 'ot':
            # Use the OpenTopography Enterprise API to download the point cloud data 
        
            # Ensure base_url matches your API's base URL
            base_url = "https://portal.opentopography.org//API"
            endpoint = "/pointcloud"
            
            params = {
            "datasetName": dataset_id,
            "south" : self.da.bounds['south'], 
            "north" : self.da.bounds['north'], 
            "west" : self.da.bounds['west'], 
            "east" : self.da.bounds['east'], 
            "API_Key" : API_Key, #All OT hosted point cloud datasets require an enterprise partner API key for access. Please email info@opentopography.org for more information.   
        }
        
            # Make the GET request to the API
            response = requests.get(url=base_url + endpoint, params=params)

            # Check if the request was successful
            if response.status_code == 200:
                filename = folder + output_name+'_'+dataset_type+'.laz'
                with open(filename, 'wb') as file:
                    file.write(response.content)
            # Wait until the file is fully downloaded
                while not os.path.exists(filename):
                    time.sleep(1)
            
            
                ot_dtm_pipeline = self.make_DEM_pipeline_from_file(folder+output_name+'_'+dataset_type+'.laz', bounds_polygon_epsg_initial_crs, dem_resolution,
                                    filterNoise=False, reclassify=False, savePointCloud=False, outCRS=dataset_crs,
                                    pc_outName=folder+output_name, pc_outType='laz', demType='dtm', gridMethod='idw', 
                                    dem_outName=folder+output_name+'_'+dataset_type+'_DTM', dem_outExt='tif', driver="GTiff")
                ot_dtm_pipeline = pdal.Pipeline(json.dumps(ot_dtm_pipeline))
                ot_dtm_pipeline.execute_streaming(chunk_size=1000000)

                ot_dsm_pipeline = self.make_DEM_pipeline_from_file(folder+output_name+'_'+dataset_type+'.laz', bounds_polygon_epsg_initial_crs, dem_resolution,
                                        filterNoise=False, reclassify=False, savePointCloud=False, outCRS=dataset_crs,
                                        pc_outName=folder+output_name, pc_outType='laz', demType='dsm', gridMethod='max', 
                                        dem_outName=folder+output_name+'_'+dataset_type+'_DSM', dem_outExt='tif', driver="GTiff")
                ot_dsm_pipeline = pdal.Pipeline(json.dumps(ot_dsm_pipeline))
                ot_dsm_pipeline.execute_streaming(chunk_size=1000000)
                
            else:
                print(f"Error: {response.status_code}")
        
        elif data_source_ == "usgs":
            usgs_dtm_pipeline = self.make_DEM_pipeline_aws(bounds_polygon_epsg_initial_crs, [dataset_id], pc_resolution, dem_resolution, data_source = "usgs",
                    filterNoise = False, reclassify = False, savePointCloud = False, outCRS = dataset_crs,
                    pc_outName = folder+output_name+'_'+dataset_type, pc_outType = 'laz', demType = 'dtm', gridMethod = 'idw', 
                    dem_outName = folder+output_name+'_'+dataset_type+'_DTM', dem_outExt = 'tif', driver = "GTiff")

            usgs_dtm_pipeline = pdal.Pipeline(json.dumps(usgs_dtm_pipeline))
            usgs_dtm_pipeline.execute_streaming(chunk_size=1000000)
            
            usgs_dsm_pipeline = self.make_DEM_pipeline_aws(bounds_polygon_epsg_initial_crs, [dataset_id], pc_resolution, dem_resolution, data_source = "usgs",
                            filterNoise = False, reclassify = False, savePointCloud = False, outCRS = dataset_crs,
                            pc_outName = folder+output_name+'_'+dataset_type, pc_outType = 'laz', demType = 'dsm', gridMethod = 'max', 
                            dem_outName = folder+output_name+'_'+dataset_type+'_DSM', dem_outExt = 'tif', driver = "GTiff")
            
            usgs_dsm_pipeline = pdal.Pipeline(json.dumps(usgs_dsm_pipeline))
            usgs_dsm_pipeline.execute_streaming(chunk_size=1000000)
        
        elif data_source_ == "noaa":
            # Use AWS bucket and PDAL to download the point cloud data
            
            noaa_dtm_pipeline = self.make_DEM_pipeline_aws(bounds_polygon_epsg_initial_crs, [dataset_id], pc_resolution, dem_resolution, data_source = "noaa",
                            filterNoise = False, reclassify = False, savePointCloud = False, outCRS = dataset_crs,
                            pc_outName = folder+output_name+'_'+dataset_type, pc_outType = 'laz', demType = 'dtm', gridMethod = 'idw', 
                            dem_outName = folder+output_name+'_'+dataset_type+'_DTM', dem_outExt = 'tif', driver = "GTiff")
            noaa_dtm_pipeline = pdal.Pipeline(json.dumps(noaa_dtm_pipeline))
            noaa_dtm_pipeline.execute_streaming(chunk_size=1000000)
            
            noaa_dsm_pipeline = self.make_DEM_pipeline_aws(bounds_polygon_epsg_initial_crs, [dataset_id], pc_resolution, dem_resolution, data_source = "noaa",
                              filterNoise = False, reclassify = False, savePointCloud = False, outCRS = dataset_crs,
                              pc_outName = folder+output_name+'_'+dataset_type, pc_outType = 'laz', demType = 'dsm', gridMethod = 'max', 
                              dem_outName = folder+output_name+'_'+dataset_type+'_DSM', dem_outExt = 'tif', driver = "GTiff")
            noaa_dsm_pipeline = pdal.Pipeline(json.dumps(noaa_dsm_pipeline))
            noaa_dsm_pipeline.execute_streaming(chunk_size=1000000)
        
        else:
            raise ValueError("Data source must be either 'ot', 'usgs' or 'noaa'")
        
        # Determine which DEM paths to fill
        if dataset_type == "compare":
            dtm_path = self.compare_dtm_path
            dsm_path = self.compare_dsm_path
        else:
            dtm_path = self.reference_dtm_path
            dsm_path = self.reference_dsm_path

        # Fill NoData holes in the generated DEMs
        self.fill_no_data(dtm_path, dtm_path, method=method, nodata=nodata, max_dist=max_dist, smooth_iter=smooth_iter)
        self.fill_no_data(dsm_path, dsm_path, method=method, nodata=nodata, max_dist=max_dist, smooth_iter=smooth_iter)
        
    # def fill_dems(self):
        
    
    # def raster_files(self):
    #     self.raster_compare_dtm = Raster(self.compare_dtm_path)
    #     self.raster_compare_dsm = Raster(self.compare_dsm_path)
    #     self.raster_reference_dtm = Raster(self.reference_dtm_path)
    #     self.raster_reference_dsm = Raster(self.reference_dsm_path)
    #     self.raster_pair_compare_dtm_dsm = RasterPair(raster1 = self.compare_dtm_path, raster2 = self.compare_dsm_path)
    #     self.raster_pair_reference_dtm_dsm = RasterPair(raster1 = self.reference_dtm_path, raster2 = self.reference_dsm_path)
    #     self.raster_pair_topo_diff_dtm = RasterPair(raster1 = self.compare_dtm_path, raster2 = self.reference_dtm_path)
    #     self.raster_pair_topo_diff_dsm = RasterPair(raster1 = self.compare_dsm_path, raster2 = self.reference_dsm_path)

"""Geoid-handling utilities – cross‑platform version
=================================================

This module equips the Topographic Processing Toolkit with **automatic
vertical geoid alignment**.  It resolves, downloads, and applies the correct
geoid grid for *any* model available on `https://cdn.proj.org`, not just the
U.S. NOAA set.

Key improvements (v3)
--------------------
* **Robust CDN index fetch** – gracefully handles network / JSON errors and
  falls back to direct `token.tif` probes when `index.json` is unavailable.
* Explicit **`requests.head` existence check** so we know a TIFF really
  exists before returning its URL.
* Tidier exception messages that bubble up the root cause for easier
  debugging in notebooks.

"""



__all__ = ["GeoidTransformer"]

# ----------------------------------------------------------------------
# Constants & alias tables
# ----------------------------------------------------------------------
_CDN_ROOT = "https://cdn.proj.org"
_CDN_INDEX_URL = f"{_CDN_ROOT}/index.json"
_CDN_CACHE = Path(tempfile.gettempdir()) / "proj_cdn_index.json"
_CDN_INDEX_TTL_H = 24  # hours

_GEOID_ALIAS_MAP: Dict[str, str] = {
    # United States ----------------------------------------------------
    # GEOID18 / 2018
    "geoid2018": "us_noaa_g2018u0",
    "geoid 2018": "us_noaa_g2018u0",
    "geoid18": "us_noaa_g2018u0",
    "geoid 18": "us_noaa_g2018u0",
    "g2018": "us_noaa_g2018u0",

    # GEOID12 (single file – 2012b – served as g2012bu0)
    "geoid12": "us_noaa_g2012bu0",
    "geoid 12": "us_noaa_g2012bu0",
    "geoid12a": "us_noaa_g2012bu0",
    "geoid12b": "us_noaa_g2012bu0",
    "geoid 12a": "us_noaa_g2012bu0",
    "geoid 12b": "us_noaa_g2012bu0",
    "geoid2012": "us_noaa_g2012bu0",
    "geoid 2012": "us_noaa_g2012bu0",

    # GEOID09
    "geoid09": "us_noaa_geoid09_conus",
    "geoid 09": "us_noaa_geoid09_conus",
    "geoid9": "us_noaa_geoid09_conus",
    "geoid 9": "us_noaa_geoid09_conus",
    "geoid2009": "us_noaa_geoid09_conus",
    "geoid 2009": "us_noaa_geoid09_conus",

    # GEOID06
    "geoid06": "us_noaa_geoid06_conus",
    "geoid 06": "us_noaa_geoid06_conus",
    "geoid6": "us_noaa_geoid06_conus",
    "geoid 6": "us_noaa_geoid06_conus",
    "geoid2006": "us_noaa_geoid06_conus",
    "geoid 2006": "us_noaa_geoid06_conus",

    # GEOID03
    "geoid03": "us_noaa_geoid03_conus",
    "geoid 03": "us_noaa_geoid03_conus",
    "geoid3": "us_noaa_geoid03_conus",
    "geoid 3": "us_noaa_geoid03_conus",
    "geoid2003": "us_noaa_geoid03_conus",
    "geoid 2003": "us_noaa_geoid03_conus",

    # Australia --------------------------------------------------------
    "ausgeoid2020": "au_ga_ausgeoid2020",
    "ausgeoid09": "au_ga_ausgeoid09",

    # Canada -----------------------------------------------------------
    "ncg2013a": "ca_nrc_ncg2013a",

    # France -----------------------------------------------------------
    "rgf93vg18": "fr_ign_rgf93vg18",
}

_GEOID_RE = re.compile(r"geoid\s*([0-9]{2,4}|[a-z]+)", re.I)
_DATUM_RE = re.compile(r"(NAVD88|NGVD29|AHD|EVRF|EGM96|EGM2008)", re.I)

# ----------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------

def _safe_get_json(url: str, timeout: int = 30) -> Optional[dict | list]:
    """Return JSON from *url*, or ``None`` if decoding fails."""
    try:
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except (requests.RequestException, json.JSONDecodeError):
        return None


def _fetch_cdn_index() -> List[dict]:
    #"""Load and cache cdn.proj.org/index.json (graceful on failure)."""
    if _CDN_CACHE.exists() and (time.time() - _CDN_CACHE.stat().st_mtime < _CDN_INDEX_TTL_H * 3600):
        try:
            return json.loads(_CDN_CACHE.read_text())
        except json.JSONDecodeError:
            _CDN_CACHE.unlink(missing_ok=True)

    idx_json = _safe_get_json(_CDN_INDEX_URL)
    if idx_json is not None:
        _CDN_CACHE.write_text(json.dumps(idx_json))
        return idx_json  # type: ignore[return-value]

    warnings.warn("Could not fetch cdn.proj.org index – falling back to direct file probing.")
    return []


def _bbox_intersects(a: Tuple[float, float, float, float], b: List[float]) -> bool:
    """Shapely-free quick bbox intersection in lon/lat."""
    west1, south1, east1, north1 = a
    west2, south2, east2, north2 = b
    return not (east1 < west2 or east2 < west1 or north1 < south2 or north2 < south1)

# ----------------------------------------------------------------------
# GeoidTransformer class
# ----------------------------------------------------------------------
class GeoidTransformer:
    #"""Ensure two rasters share the same geoid model by converting raster1.”""

    def __init__(
        self,
        pair: RasterPair,
        compare_vcrs: str,
        reference_vcrs: str,
        out_dir: Optional[str | Path] = None,
    ) -> None:
        self.pair = pair
        self.compare_datum, self.compare_token = self._parse_vcrs(compare_vcrs or "")
        self.reference_datum, self.reference_token = self._parse_vcrs(reference_vcrs or "")
        self.out_dir = Path(out_dir) if out_dir else pair.raster1.path.parent

    # ------------------------------------------------------------------
    # Public driver
    # ------------------------------------------------------------------
    def ensure_common_geoid(self) -> RasterPair:
        #"""Return a RasterPair where raster1 matches raster2’s geoid.”""
        if self._needs_conversion():
            new_r1 = self._convert_raster1()
            return RasterPair(new_r1, self.pair.raster2)
        return self.pair

    # ------------------------------------------------------------------
    # Parsing helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _canonicalise(token: str | None) -> Optional[str]:
        if token is None:
            return None
        t = token.lower().strip()
        return _GEOID_ALIAS_MAP.get(t, t)

    def _parse_vcrs(self, vcrs: str) -> Tuple[Optional[str], Optional[str]]:
        datum = (_DATUM_RE.search(vcrs) or [None])[0]
        geoid_raw_match = _GEOID_RE.search(vcrs)
        geoid = self._canonicalise(geoid_raw_match.group(0) if geoid_raw_match else None)
        return datum.upper() if datum else None, geoid

    # ------------------------------------------------------------------
    def _needs_conversion(self) -> bool:
        if self.compare_datum != self.reference_datum or self.compare_datum is None:
            raise ValueError("Vertical datums differ or are undefined – cannot transform safely.")
        if self.compare_token is None or self.reference_token is None:
            raise ValueError("Missing geoid information on one or both rasters.")
        return self.compare_token != self.reference_token

    # ------------------------------------------------------------------
    # Grid resolution logic
    # ------------------------------------------------------------------
    def _resolve_grid_base(self, token: str) -> str:
        #"""Return *filename* (without extension) of the best‑fit grid.”""
        # 1. Try alias table (direct hit)
        if token in _GEOID_ALIAS_MAP.values():
            return token  # already full basename

        idx = _fetch_cdn_index()
        if idx:
            # AOI in lon/lat
            with rasterio.open(self.pair.raster1.path) as src:
                left, bottom, right, top = src.bounds
                aoi_ll = Transformer.from_crs(src.crs, "EPSG:4326", always_xy=True).transform_bounds(left, bottom, right, top)

            # Candidates whose names contain token
            cands = [item for item in idx if token in item["name"]]
            # Prefer ones intersecting AOI (via .properties)
            for item in cands:
                props = _safe_get_json(f"{_CDN_ROOT}/{item['name']}.properties", timeout=15)
                if props and _bbox_intersects(aoi_ll, props.get("bounds", [-180, -90, 180, 90])):
                    return item["name"]
            if cands:
                return cands[0]["name"]

        # 2. Fallback: assume token is the base name
        return token

    def _grid_url(self, token: str) -> str:
        base = self._resolve_grid_base(token)
        url = f"{_CDN_ROOT}/{base}.tif"
        # Verify existence quickly
        try:
            if requests.head(url, timeout=10).ok:
                return url
        except requests.RequestException:
            pass
        raise ValueError(f"Could not locate grid for geoid token '{token}'. Tried {url}")

    # ------------------------------------------------------------------
    def _download_grid(self, token: str) -> Path:
        url = self._grid_url(token)
        cache_dir = Path(tempfile.gettempdir()) / "topotoolkit_geoids"
        cache_dir.mkdir(parents=True, exist_ok=True)
        out = cache_dir / Path(url).name
        if out.exists():
            return out
        with requests.get(url, stream=True, timeout=60) as r:
            r.raise_for_status()
            with open(out, "wb") as fp:
                for chunk in r.iter_content(1 << 20):
                    fp.write(chunk)
        return out

    # ------------------------------------------------------------------
    @staticmethod
    def _aligned_grid(grid_path: Path, target: Raster) -> np.ndarray:
        da = rio.open_rasterio(grid_path, masked=True)
        return da.rio.reproject_match(target.data).squeeze().values

    # ------------------------------------------------------------------
    def _convert_raster1(self) -> Raster:
        r1 = self.pair.raster1
        g1 = self._aligned_grid(self._download_grid(self.compare_token), r1)
        g2 = self._aligned_grid(self._download_grid(self.reference_token), r1)

        # Retrieve DEM as a (possibly) masked numpy array
        dem = r1.data.squeeze().values  # may be a np.ma.MaskedArray
        mask = np.ma.getmaskarray(dem)  # returns False array if no mask present

        converted = np.where(~mask, dem - g1 + g2, np.nan)

        out_fname = f"{r1.path.stem}_geo_{self.compare_token}_to_{self.reference_token}.tif"
        out_path = self.out_dir / out_fname
        with rasterio.open(r1.path) as src:
            profile = src.profile.copy()
            profile.update(dtype="float32", nodata=np.nan)
            with rasterio.open(out_path, "w", **profile) as dst:
                dst.write(converted.astype("float32"), 1)

        return Raster(out_path)


# ----------------------------------------------------------------------
# Terrain derivatives                    
# ----------------------------------------------------------------------
class TerrainDerivatives:
    """Compute hillshade, slope, aspect, and roughness via GDAL DEMProcessing."""

    def __init__(self, out_dir: Path | str):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def _process(self, src: Path, name: str, mode: str, options=None, **kwargs) -> Path:
        dst = self.out_dir / name
        if not dst.exists():
            if options is not None:
                # only pass options if you actually have one
                gdal.DEMProcessing(str(dst), str(src), mode, options=options, **kwargs)
            else:
                # otherwise call without the options kwarg
                gdal.DEMProcessing(str(dst), str(src), mode, **kwargs)
        return dst

    def hillshade(self, dem: Path, azimuth: float = 315, altitude: float = 45) -> Path:
        opts = gdal.DEMProcessingOptions(azimuth=azimuth, altitude=altitude)
        return self._process(dem, f"{Path(dem).stem}_hillshade.tif", "hillshade", options=opts)

    def slope(self, dem: Path) -> Path:
        return self._process(dem, f"{Path(dem).stem}_slope.tif", "slope")

    def aspect(self, dem: Path) -> Path:
        try:
            opts = gdal.DEMProcessingOptions(zeroForFlat=True)
        except TypeError:  # very old GDAL
            warnings.warn("GDAL < 3.4 lacks zeroForFlat – flat areas will be 0")
            opts = None
        return self._process(dem, f"{Path(dem).stem}_aspect.tif", "aspect", options=opts)

    def roughness(self, dem: Path) -> Path:
        return self._process(dem, f"{Path(dem).stem}_roughness.tif", "roughness")

# ----------------------------------------------------------------------
# TopoDifferencer                                                
# ----------------------------------------------------------------------
class TopoDifferencer:
    """Compute raster differences, NoData masks, and export/plot results."""

    def __init__(self, out_dir: Path | str):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _sym_range(arr: np.ndarray):
        absmax = np.nanmax(np.abs(arr))
        return -absmax, absmax

    @staticmethod
    def _extent(raster: "Raster"):
        left, bottom, right, top = rasterio.open(raster.path).bounds
        return [left, right, bottom, top]

    def difference_da(self, pair: RasterPair):
        """Return an xarray DataArray of raster2 − raster1 (masked)."""
        return (pair.raster2_data - pair.raster1_data).compute()

    def save_difference_raster(self, pair: RasterPair, filename: str) -> Raster:
        diff_da = self.difference_da(pair)
        out = self.out_dir / f"{filename}.tif"
        diff_da.rio.to_raster(out)
        return Raster(out)

    def combined_mask(
        self,
        *,
        pair: RasterPair | None = None,
        a: Path | str | None = None,
        b: Path | str | None = None,
        name: str = "mask.tif",
    ) -> Raster:
        if pair is not None and (a or b):
            raise ValueError("Provide either 'pair' or 'a' & 'b', not both")
        if pair is None and None in (a, b):
            raise ValueError("Need both 'a' and 'b' paths when pair is not given")
        if pair is not None:
            r1, r2 = pair.raster1, pair.raster2
        else:
            r1, r2 = Raster(a), Raster(b)
            if r1.shape != r2.shape or r1.crs != r2.crs:
                r2 = r2.reproject_to(r1)
        mask = np.logical_or(r1.data.mask.squeeze(), r2.data.mask.squeeze())
        with rasterio.open(r1.path) as src:
            meta = src.meta.copy()
        meta.update(dtype="uint8", count=1, nodata=0)
        out = self.out_dir / name
        with rasterio.open(out, "w", **meta) as dst:
            dst.write(mask.astype("uint8"), 1)
        return Raster(out)

    def plot_difference(
        self,
        *,
        pair: RasterPair | None = None,
        diff_path: Path | str | None = None,
        overlay: Raster | None = None,
        mask_overlay: bool = True,
        cmap="RdBu_r",
        vmin: float | None = None,
        vmax: float | None = None,
        center_zero: bool = True,
        overlay_alpha: float = 0.4,
        title: str = "Difference",
        save_path: Path | str | None = None,
        dpi: int = 300,
    ) -> plt.Figure:
        """Plot (and optionally save) a difference raster with hillshade overlay.

        If *overlay* is provided it is drawn in grayscale beneath the diff map
        (alpha-blended). A combined NoData mask is applied when
        *mask_overlay* is True so only pixels valid in **both** rasters are
        shown.
        """
        if (pair is None) == (diff_path is None):
            raise ValueError("Provide exactly one of 'pair' or 'diff_path'")

        # Obtain diff DataArray and drop singleton band dimension
        if pair is not None:
            diff_da = self.difference_da(pair)
            diff_da = diff_da.squeeze(dim=[d for d in diff_da.dims if diff_da[d].size == 1], drop=True)
            base = pair.raster1
        else:
            diff_r = Raster(diff_path)
            # raster data has shape (bands, y, x) or (y, x)
            diff_da = diff_r.data.squeeze()
            base = diff_r

        # Convert to numpy and ensure 2D
        diff_arr = np.squeeze(diff_da.values)
        extent = self._extent(base)

        # Determine colour limits
        if center_zero and vmin is None and vmax is None:
            vmin, vmax = self._sym_range(diff_arr)

        # Prepare overlay alignment & masking
        if overlay is not None:
            ov = overlay
            if ov.shape[1:] != base.shape[1:] or ov.crs != base.crs:
                ov = ov.reproject_to(base)
            ov_arr = np.squeeze(ov.data.values)
            if mask_overlay:
                valid = ~np.logical_or(
                    np.isnan(diff_arr), np.isnan(ov_arr)
                )
                diff_arr = np.where(valid, diff_arr, np.nan)
                ov_arr = np.where(valid, ov_arr, np.nan)
        else:
            ov_arr = None

        # --- plotting ---
        fig, ax = plt.subplots(figsize=(10, 6))
        cmap_obj = plt.get_cmap(cmap)
        cmap_obj.set_bad(color="none")
        norm = plt.Normalize(vmin=vmin, vmax=vmax)
        im = ax.imshow(diff_arr, cmap=cmap_obj, norm=norm, extent=extent)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        if ov_arr is not None:
            shade_cmap = plt.get_cmap("gray")
            shade_cmap.set_bad(color="none")
            ax.imshow(ov_arr, cmap=shade_cmap, alpha=overlay_alpha, extent=extent)
        ax.axis("off")
        ax.set_title(title)

        if save_path is not None:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=dpi, bbox_inches="tight", pad_inches=0)
        return fig
    
    from pathlib import Path


def get_colormap_bounds(arr: np.ndarray) -> tuple[float, float]:
    """
    Compute symmetric color bounds around zero for a difference array.
    """
    valid = ~np.isnan(arr)
    if not np.any(valid):
        return 0.0, 0.0
    v = np.abs(arr[valid])
    m = v.max()
    return -m, m


def reproject_for_map(src_path: Path | str, dst_path: Path | str, dst_crs: str = "EPSG:3857"):
    """
    Reproject a raster from its current CRS to a destination CRS.
    """
    with rio.open_rasterio(src_path, masked=True) as src:
        da = src.rio.reproject(dst_crs)
        da.rio.to_raster(dst_path)
        

