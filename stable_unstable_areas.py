from pathlib import Path
import statistics
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
import geopandas as gpd
from shapely.geometry import shape, mapping, Polygon
from shapely.ops import transform as shapely_transform
from pyproj import Transformer
import matplotlib.pyplot as plt
from ipyleaflet import Map, DrawControl, ImageOverlay, GeoJSON, LegendControl, WidgetControl
from ipywidgets import Button, HBox, Label
from scipy import stats

from differencing_functions import Raster

class TopoMapInteractor:
    """
    Interactive map for drawing 'stable' and 'unstable' polygons on a topo-difference raster,
    with pixel-count utility, two-layer legend, and labeled draw buttons.
    """
    def __init__(
        self,
        topo_diff_path: Path | str,
        hillshade_path: Path | str,
        output_dir: Path | str,
        zoom: int = 15,
        map_size: tuple[str, str] = ('800px', '1300px'),
        overlay_cmap: str = 'bwr_r',
        overlay_dpi: int = 300
    ):
        # Load rasters
        self.topo_diff = Raster(topo_diff_path)
        self.hillshade = Raster(hillshade_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Storage for geometries
        self.stable_geoms: list[Polygon] = []
        self.unstable_geoms: list[Polygon] = []
        self.current_category: str | None = None

        # Compute lat/lon bounds
        with rasterio.open(self.topo_diff.path) as ds:
            bounds = ds.bounds
            crs = ds.crs
        transformer = Transformer.from_crs(crs, 'EPSG:4326', always_xy=True)
        west, south = transformer.transform(bounds.left, bounds.bottom)
        east, north = transformer.transform(bounds.right, bounds.top)
        self.latlon_bounds = ((south, west), (north, east))

        # Generate overlay PNG
        png_path = self.output_dir / f"{Path(self.topo_diff.path).stem}.png"
        self._generate_overlay_png(png_path, cmap=overlay_cmap, dpi=overlay_dpi)

        # Initialize map
        center = ((north + south) / 2, (west + east) / 2)
        self.map = Map(
            center=center,
            zoom=zoom,
            layout={'height': map_size[0], 'width': map_size[1]}
        )

        # Add image overlay
        self.map.add_layer(ImageOverlay(url=str(png_path), bounds=self.latlon_bounds))

        # Legend
        legend_dict = {'Stable Area': 'green', 'Unstable Area': 'red'}
        self.map.add_control(LegendControl(legend_dict, title='Legend'))

        # GeoJSON layers
        self.geojson_stable = GeoJSON(data={"type": "FeatureCollection", "features": []},
                                     style={"color": "green", "fillColor": "green", "fillOpacity": 0.3})
        self.geojson_unstable = GeoJSON(data={"type": "FeatureCollection", "features": []},
                                       style={"color": "red", "fillColor": "red", "fillOpacity": 0.3})
        self.map.add_layer(self.geojson_stable)
        self.map.add_layer(self.geojson_unstable)

        # Single DrawControl reused
        self.draw_control = DrawControl(
            polygon={"shapeOptions": {"weight": 2, "fillOpacity": 0.3}},
        )
        # disable other shapes
        for attr in ('circle', 'circlemarker', 'polyline', 'rectangle'):
            setattr(self.draw_control, attr, {})
        self.draw_control.on_draw(self._handle_draw)

        # Create labeled buttons
        self.btn_stable = Button(description='Stable', layout={'width': '80px'})
        self.btn_unstable = Button(description='Unstable', layout={'width': '80px'})
        # style colors
        self.btn_stable.style.button_color = 'lightgreen'
        self.btn_unstable.style.button_color = 'lightcoral'

        # Button callbacks
        self.btn_stable.on_click(lambda _: self._activate_category('stable'))
        self.btn_unstable.on_click(lambda _: self._activate_category('unstable'))

        # Place buttons above map
        btn_box = HBox([Label(' Draw mode:'), self.btn_stable, self.btn_unstable])
        self.map.add_control(WidgetControl(widget=btn_box, position='topright'))

    def _generate_overlay_png(self, png_path, cmap='bwr_r', dpi=300):
        arr = np.squeeze(self.topo_diff.data.values)
        vmin, vmax = self._get_sym_bounds(arr)
        fig, ax = plt.subplots(frameon=False)
        ax.axis('off')
        cmap_obj = plt.get_cmap(cmap); cmap_obj.set_bad(color='none')
        ax.imshow(arr, cmap=cmap_obj, vmin=vmin, vmax=vmax)
        fig.savefig(png_path, dpi=dpi, bbox_inches='tight', pad_inches=0)
        plt.close(fig)

    @staticmethod
    def _get_sym_bounds(arr: np.ndarray) -> tuple[float, float]:
        flat = arr.flatten(); valid = flat[~np.isnan(flat)]
        return (-0.0, 0.0) if valid.size == 0 else (-np.max(np.abs(valid)), np.max(np.abs(valid)))

    def _activate_category(self, category: str):
        """Enable drawing for specified category."""
        self.current_category = category
        if self.draw_control not in self.map.layers:
            self.map.add_control(self.draw_control)
        # highlight active button
        self.btn_stable.style.font_weight = 'bold' if category=='stable' else 'normal'
        self.btn_unstable.style.font_weight = 'bold' if category=='unstable' else 'normal'

    def _handle_draw(self, control, action, geo_json):
        """Handles polygon draw, assigns based on current_category."""
        if action!='created' or geo_json['geometry']['type']!='Polygon': return
        geom = shape(geo_json['geometry'])
        if self.current_category=='stable':
            self.stable_geoms.append(geom)
            feats=[mapping(g) for g in self.stable_geoms]
            self.geojson_stable.data={"type":"FeatureCollection","features":feats}
        elif self.current_category=='unstable':
            self.unstable_geoms.append(geom)
            feats=[mapping(g) for g in self.unstable_geoms]
            self.geojson_unstable.data={"type":"FeatureCollection","features":feats}
        # remove draw control and reset
        self.map.remove_control(self.draw_control)
        self.current_category=None
        self.btn_stable.style.font_weight='normal'; self.btn_unstable.style.font_weight='normal'

    def calculate_pixel_count(self, polygon: Polygon) -> int:
        """
        Returns count of valid pixels inside a polygon (assumed in lat/lon), reprojected to raster CRS.
        """
        with rasterio.open(self.topo_diff.path) as src:
            dst_crs = src.crs
            # transform polygon from EPSG:4326 to raster CRS
            transformer = Transformer.from_crs('EPSG:4326', dst_crs, always_xy=True)
            poly_proj = shapely_transform(transformer.transform, polygon)
            # mask with projected polygon
            out_img, _ = rasterio.mask.mask(src, [mapping(poly_proj)], crop=True)
            data = out_img[0]; nodata = src.nodata
            valid = (~np.isnan(data)) & (data != nodata)
            return int(np.sum(valid))

    def get_geodataframes(self) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
        """
        Return two GeoDataFrames in the raster's CRS for stable and unstable polygons.

        Each GeoDataFrame includes:
        - 'Name': polygon identifier
        - 'geometry': polygon geometry in raster CRS
        - 'pixels': count of valid (non-nodata) raster pixels covered by each polygon
        """
        # Build GeoDataFrames in lat/lon CRS
        gdf_stable = gpd.GeoDataFrame(
            {'Name': [f'stable{i+1}' for i in range(len(self.stable_geoms))],
             'geometry': self.stable_geoms,
             'pixels': [self.calculate_pixel_count(g) for g in self.stable_geoms]
             },
            crs='EPSG:4326'
        )
        gdf_unstable = gpd.GeoDataFrame(
            {'Name': [f'unstable{i+1}' for i in range(len(self.unstable_geoms))],
             'geometry': self.unstable_geoms,
             'pixels': [self.calculate_pixel_count(g) for g in self.unstable_geoms]
             },
            crs='EPSG:4326'
        )
        # Reproject to raster CRS
        with rasterio.open(self.topo_diff.path) as src:
            raster_crs = src.crs
        gdf_stable = gdf_stable.to_crs(raster_crs)
        gdf_unstable = gdf_unstable.to_crs(raster_crs)
        return gdf_stable, gdf_unstable

def descriptive_stats(values: np.ndarray) -> pd.DataFrame:
    """
    Compute descriptive statistics for a 1D array of values.
    Returns a one-row DataFrame.
    """
    data = values[~np.isnan(values)]
    if data.size == 0:
        # return empty stats row
        cols = ['mean','median','mode','std','variance','min','max',
                'skewness','kurtosis','0.5_percentile','99.5_percentile']
        return pd.DataFrame([{c: np.nan for c in cols}])
    mean = np.mean(data)
    median = np.median(data)
    try:
        mode = statistics.mode(data)
    except statistics.StatisticsError:
        mode = np.nan
    std = np.std(data)
    var = np.var(data)
    minimum = np.min(data)
    maximum = np.max(data)
    skew = stats.skew(data)
    kurt = stats.kurtosis(data)
    p1, p99 = np.percentile(data, [0.5, 99.5])
    return pd.DataFrame([{
        'mean': mean,
        'median': median,
        'mode': mode,
        'std': std,
        'variance': var,
        'min': minimum,
        'max': maximum,
        'skewness': skew,
        'kurtosis': kurt,
        '0.5_percentile': p1,
        '99.5_percentile': p99
    }])

class StableAreaRasterizer:
    """
    Rasterize stable-area polygons to mask a topographic-difference raster:
    - `rasterize_all`: one output where inside polygons = original values, outside = nodata.
    - `rasterize_each`: separate rasters per polygon.
    """
    def __init__(self, topo_diff_path: Path | str, stable_gdf, nodata: float = -9999):
        self.topo_path = Path(topo_diff_path)
        self.gdf = stable_gdf.copy()
        self.nodata = nodata

    def rasterize_all(self, output_path: Path | str) -> Path:
        """
        Create a single GeoTIFF where values inside any stable polygon are
        preserved, and outside are set to `nodata`.
        """
        out_path = Path(output_path)
        with rasterio.open(self.topo_path) as src:
            profile = src.profile.copy()
            profile.update(nodata=self.nodata)
            data = src.read(1)
            # rasterize mask of polygons
            mask = rasterize(
                [(geom, 1) for geom in self.gdf.geometry],
                out_shape=src.shape,
                transform=src.transform,
                fill=0,
                dtype='uint8'
            )
            # apply mask
            out = np.where(mask == 1, data, self.nodata)
            # write
            with rasterio.open(out_path, 'w', **profile) as dst:
                dst.write(out, 1)
        return out_path

    def rasterize_each(self, output_dir: Path | str) -> dict[int, Path]:
        """
        For each polygon in the GeoDataFrame, write a GeoTIFF with values
        inside that polygon preserved and outside = `nodata`.
        Returns a dict mapping area_id -> raster path.
        """
        outdir = Path(output_dir)
        outdir.mkdir(exist_ok=True, parents=True)
        paths = {}
        with rasterio.open(self.topo_path) as src:
            profile = src.profile.copy()
            profile.update(nodata=self.nodata)
            data = src.read(1)
            for idx, row in self.gdf.iterrows():
                geom = row.geometry
                mask = rasterize(
                    [(geom, 1)],
                    out_shape=src.shape,
                    transform=src.transform,
                    fill=0,
                    dtype='uint8'
                )
                out = np.where(mask == 1, data, self.nodata)
                path = outdir / f"stable_area_{idx}.tif"
                with rasterio.open(path, 'w', **profile) as dst:
                    dst.write(out, 1)
                paths[idx] = path
        return paths

class StableAreaAnalyzer:
    """
    Use rasters produced by StableAreaRasterizer to compute descriptive stats:
    - `stats_all`: stats on combined-area raster.
    - `stats_each`: stats on each individual-area raster.
    """
    def __init__(self, rasterizer: StableAreaRasterizer):
        self.rasterizer = rasterizer

    def _stats_from_raster(self, path: Path | str) -> pd.DataFrame:
        with rasterio.open(path) as src:
            arr = src.read(1)
            arr = np.where(arr == src.nodata, np.nan, arr)
            flat = arr.ravel()
        return descriptive_stats(flat)

    def stats_all(self, output_path: Path | str) -> pd.DataFrame:
        """
        Rasterize all polygons, write to `output_path`, and return one-row stats DataFrame.
        """
        out = self.rasterizer.rasterize_all(output_path)
        df = self._stats_from_raster(out)
        df.index = ['all_areas']
        return df

    def stats_each(self, output_dir: Path | str) -> pd.DataFrame:
        """
        Rasterize each polygon into `output_dir`, compute stats per area,
        and return DataFrame indexed by area_id.
        """
        paths = self.rasterizer.rasterize_each(output_dir)
        records = []
        for area_id, path in paths.items():
            df = self._stats_from_raster(path)
            df['area_id'] = area_id
            records.append(df)
        result = pd.concat(records, ignore_index=True).set_index('area_id')
        return result
