import math
from typing import Sequence, Optional, Dict, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import curve_fit
import rasterio
import rioxarray as rio
from numba import njit, prange


def dropna(array):
    """
    Drops NaN values from a NumPy array.

    Parameters
    ----------
    array : np.ndarray
        The input NumPy array from which to drop NaN values.

    Returns
    -------
    np.ndarray
        A new array with NaN values removed.
    """
    return array[~np.isnan(array)]

class RasterDataHandler:
    """
    A class used for loading vertical differencing raster data, 
    subtracting a vertical systematic error from the raster,
    and randomly sampling the raster data for further analysis.

    Attributes
    ----------
    raster_path : str
        The file path to the raster data.
    unit : str
        The unit of measurement for the raster data (for plotting labels).
    resolution : float
        The resolution of the raster data.
    rioxarray_obj : rioxarray.DataArray or None
        The rioxarray object holding the raster data.
    data_array : numpy.ndarray or None
        The loaded raster data as a 1D NumPy array (NaNs removed).
    transformed_values : numpy.ndarray or None
        Placeholder for any future transformations if needed.
    samples : numpy.ndarray or None
        Sampled values from the raster data.
    coords : numpy.ndarray or None
        Coordinates of the sampled values.

    Methods
    -------
    load_raster(masked=True)
        Loads the raster data from the given path, optionally applying a mask to exclude NaN values.
    subtract_value_from_raster(output_raster_path, value_to_subtract)
        Subtracts a given value from the raster data and saves the result to a new file.
    plot_raster(plot_title)
        Plots a raster image using the loaded rioxarray object.
    sample_raster(area_side, samples_per_area, max_samples)
        Samples the raster data based on a given density and maximum number of samples.
    """
    def __init__(self, raster_path, unit, resolution):
        """
        Initialize the RasterDataHandler.

        Parameters
        ----------
        raster_path : str
            The file path to the raster data.
        unit : str
            The unit of measurement for the raster data.
        resolution : float
            The resolution of the raster data.
        """
        self.raster_path = raster_path
        self.unit = unit
        self.resolution = resolution
        self.rioxarray_obj = None
        self.data_array = None
        self.transformed_values = None
        self.samples = None
        self.coords = None

    def load_raster(self, masked=True):
        """
        Loads the raster data from the specified path, applying a mask to exclude NaN values if requested.

        Parameters
        ----------
        masked : bool, optional
            If True, NaN values in the raster data will be masked (default is True).
        """
        self.rioxarray_obj = rio.open_rasterio(self.raster_path, masked=masked)
        self.data_array = self.rioxarray_obj.data[~np.isnan(self.rioxarray_obj.data)].flatten()

    def subtract_value_from_raster(self, output_raster_path, value_to_subtract):
        """
        Subtracts a specified value from the raster data and saves the resulting raster to a new file.

        Parameters
        ----------
        output_raster_path : str
            The file path where the modified raster will be saved.
        value_to_subtract : float
            The value to be subtracted from each pixel in the raster data.
        """
        with rasterio.open(self.raster_path) as src:
            data = src.read()
            nodata = src.nodata

            # Create a mask of valid data
            if nodata is not None:
                mask = data != nodata
            else:
                mask = np.ones(data.shape, dtype=bool)

            data = data.astype(float)
            data[mask] -= value_to_subtract

            out_meta = src.meta.copy()
            out_meta.update({'dtype': 'float32', 'nodata': nodata})

            with rasterio.open(output_raster_path, 'w', **out_meta) as dst:
                dst.write(data)

    def plot_raster(self, plot_title):
        """
        Plots a raster image using the rioxarray object.

        Parameters
        ----------
        plot_title : str
            The title of the plot.

        Returns
        -------
        matplotlib.figure.Figure
            The figure object containing the plot.
        """
        rio_data = self.rioxarray_obj
        fig, ax = plt.subplots(figsize=(10, 6))
        rio_data.plot(cmap="bwr_r", ax=ax, robust=True)
        ax.set_title(plot_title, pad=30)
        ax.set_xlabel('Easting (m)')
        ax.set_ylabel('Northing (m)')
        ax.ticklabel_format(style="plain")
        ax.set_aspect('equal')
        return fig

    def sample_raster(self, area_side, samples_per_area, max_samples):
        """
        Samples the raster data based on a given density and a maximum number of samples,
        returning the sampled values and their coordinates.

        The parameter 'area_side' is used as a reference to convert the cell size into
        square km if 'area_side' is given in meters (e.g., area_side=1000 for 1 km side).

        Parameters
        ----------
        area_side : float
            The reference for converting pixel area into square km (if in meters).
        samples_per_area : float
            The number of samples to draw per square kilometer.
        max_samples : int
            The maximum total number of samples to draw.

        Returns
        -------
        None
            (But populates self.samples and self.coords with the drawn sample values
            and corresponding coordinates.)
        """
        with rasterio.open(self.raster_path) as src:
            data = src.read(1)
            nodata = src.nodata
            valid_data_mask = data != nodata if nodata is not None else ~np.isnan(data)

            cell_size = src.res[0]  # Pixel size in x-direction
            # Approx. area of each pixel in "km^2" if area_side = 1000.
            cell_area_sq_km = (cell_size ** 2) / (area_side ** 2)

            # Find valid data points
            valid_data_indices = np.where(valid_data_mask)
            valid_data_count = valid_data_indices[0].size

            # Estimate total samples based on area
            total_samples = min(int(cell_area_sq_km * samples_per_area * valid_data_count), max_samples)

            if total_samples > valid_data_count:
                raise ValueError("Requested samples exceed the number of valid data points.")

            # Randomly select valid data points
            chosen_indices = np.random.choice(valid_data_count, size=total_samples, replace=False)
            rows = valid_data_indices[0][chosen_indices]
            cols = valid_data_indices[1][chosen_indices]

            # Get sampled data values
            samples = data[rows, cols]

            # Compute coordinates all at once for efficiency
            x_coords, y_coords = src.xy(rows, cols)
            coords = np.vstack([x_coords, y_coords]).T

            # Remove any NaNs
            mask = ~np.isnan(samples)
            filtered_samples = samples[mask]
            filtered_coords = coords[mask]

            self.samples = filtered_samples
            self.coords = filtered_coords

class StatisticalAnalysis:
    """
    A class to perform statistical analysis on data, including plotting data statistics
    and estimating the uncertainty of the median value of the data using bootstrap 
    resampling with subsamples of the data.

    Attributes
    ----------
    raster_data_handler : RasterDataHandler
        An instance of RasterDataHandler to manage raster data operations.

    Methods
    -------
    plot_data_stats(filtered=True)
        Plots the histogram of the raster data with exploratory statistics.
    bootstrap_uncertainty_subsample(n_bootstrap=1000, subsample_proportion=0.1)
        Estimates the uncertainty of the median value of the data using bootstrap resampling.
    """
    def __init__(self, raster_data_handler):
        """
        Initialize the StatisticalAnalysis class.

        Parameters
        ----------
        raster_data_handler : RasterDataHandler
            An instance of RasterDataHandler to manage raster data operations.
        """
        self.raster_data_handler = raster_data_handler

    def plot_data_stats(self, filtered=True):
        """
        Plots the histogram of the raster data with exploratory statistics.

        Parameters
        ----------
        filtered : bool, optional
            If True, filter the data to exclude outliers (1st and 99th percentiles) 
            for better visualization. If False, use the unfiltered data. Default is True.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The matplotlib figure object containing the histogram and statistics.

        Notes
        -----
        - The function calculates and displays the mean, median, mode(s), minimum, maximum, 
          1st quartile, and 3rd quartile of the data.
        - The histogram is plotted with 60 bins (by default).
        - The mode(s) are displayed as vertical dashed lines on the histogram.
        - A text box with the calculated statistics is added to the plot.
        """
        data = self.raster_data_handler.data_array
        if data is None or len(data) == 0:
            raise ValueError("No data available to plot. Please load the raster first.")

        mean = np.mean(data)
        median = np.median(data)
        mode_result = stats.mode(data, nan_policy='omit')
        mode_val = mode_result.mode
        q1 = np.percentile(data, 25)
        q3 = np.percentile(data, 75)
        p1 = np.percentile(data, 1)
        p99 = np.percentile(data, 99)
        minimum = np.min(data)
        maximum = np.max(data)

        # Optionally filter outliers for visualization
        if filtered:
            data = data[(data >= p1) & (data <= p99)]

        # Ensure mode_val is iterable
        if not isinstance(mode_val, np.ndarray):
            mode_val = np.array([mode_val])

        fig, ax = plt.subplots()
        # Plot histogram
        ax.hist(data, bins=60, density=False, alpha=0.6, color='g')
        ax.axvline(mean, color='r', linestyle='dashed', linewidth=1, label='Mean')
        ax.axvline(median, color='b', linestyle='dashed', linewidth=1, label='Median')
        for i, m in enumerate(mode_val):
            label = 'Mode' if i == 0 else "_nolegend_"
            ax.axvline(m, color='purple', linestyle='dashed', linewidth=1, label=label)

        # Prepare statistics text
        mode_str = ", ".join([f'{m:.3f}' for m in mode_val])
        textstr = '\n'.join((
            f'Mean: {mean:.3f}',
            f'Median: {median:.3f}',
            f'Mode(s): {mode_str}',
            f'Minimum: {minimum:.3f}',
            f'Maximum: {maximum:.3f}',
            f'1st Quartile: {q1:.3f}',
            f'3rd Quartile: {q3:.3f}'
        ))

        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=props)

        ax.set_xlabel(f'Vertical Difference ({self.raster_data_handler.unit})')
        ax.set_ylabel('Count')
        ax.set_title('Histogram of differencing results with exploratory statistics')
        ax.legend()
        plt.tight_layout()
        return fig

    def bootstrap_uncertainty_subsample(self, n_bootstrap=1000, subsample_proportion=0.1):
        """
        Estimates the uncertainty of the median value of the data using bootstrap resampling.
        This method randomly samples subsets of the data, calculates their medians, and then 
        computes the standard deviation of these medians as a measure of uncertainty.

        Parameters
        ----------
        n_bootstrap : int, optional
            The number of bootstrap samples to generate (default is 1000).
        subsample_proportion : float, optional
            The proportion of the data to include in each subsample (default is 0.1).

        Returns
        -------
        uncertainty : float
            The standard deviation of the bootstrap medians, representing 
            the uncertainty of the median value.
        """
        if (self.raster_data_handler.data_array is None or 
            len(self.raster_data_handler.data_array) == 0):
            raise ValueError("No data available for bootstrap. Please load the raster first.")

        subsample_size = int(subsample_proportion * len(self.raster_data_handler.data_array))
        bootstrap_medians = np.zeros(n_bootstrap)
        for i in range(n_bootstrap):
            sample = np.random.choice(self.raster_data_handler.data_array,
                                      size=subsample_size,
                                      replace=True)
            bootstrap_medians[i] = np.median(sample)

        return np.std(bootstrap_medians)

class VariogramAnalysis:
    """
    A class to perform variogram analysis on raster data. It calculates mean variograms,
    fits spherical models (possibly with a nugget term) to the variogram data, and plots
    the results. The code supports multiple runs to produce a mean variogram and uses
    bootstrapping to estimate parameter uncertainties.

    Attributes
    ----------
    raster_data_handler : RasterDataHandler
        An instance of RasterDataHandler to manage raster data operations.

    mean_variogram : numpy.ndarray or None
        The mean variogram calculated from multiple runs. Computed by
        `calculate_mean_variogram_numba`.

    err_variogram : numpy.ndarray or None
        The standard deviation of the variogram values across multiple runs.

    lags : numpy.ndarray or None
        The distances (lags) at which the variogram is calculated.

    mean_count : numpy.ndarray or None
        The mean count of data pairs used for each lag distance.

    best_model_config : dict or None
        A dictionary describing the best-fit model configuration:
        e.g. {"components": 2, "nugget": True}.

    fitted_variogram : numpy.ndarray or None
        The fitted variogram values (same length as `lags`) for the best-fit model.

    best_aic : float or None
        The lowest AIC value found among the tried models.

    best_params : np.ndarray or None
        The parameter array (sills, ranges, nugget if applicable) of the best-fit model.

    sills : np.ndarray or None
        The sill parameter(s) of the best model. Length equals the number of components,
        or None if pure nugget.

    ranges : np.ndarray or None
        The range parameter(s) of the best model. Length equals the number of components,
        or None if pure nugget.

    best_nugget : float or None
        The nugget effect parameter of the best model, if applicable.

    ranges_min : list or None
        The lower 2.5th percentile confidence bounds for each range (if bootstrapping succeeds).

    ranges_max : list or None
        The upper 97.5th percentile confidence bounds for each range.

    ranges_median : list or None
        The median of the bootstrapped ranges.

    sills_min : list or None
        The lower 2.5th percentile confidence bounds for each sill.

    sills_max : list or None
        The upper 97.5th percentile confidence bounds for each sill.

    sills_median : list or None
        The median of the bootstrapped sills.

    min_nugget : float or None
        The lower 2.5th percentile bound for the nugget, if present.

    max_nugget : float or None
        The upper 97.5th percentile bound for the nugget, if present.

    median_nugget : float or None
        The median bootstrapped nugget value, if present.

    param_samples : np.ndarray or None
        The array of shape (n_successful_bootstraps, n_params) containing bootstrapped parameter
        vectors from the best-fit model. May be empty or None if no bootstrap fits succeed.

    cv_mean_error_best_aic : float or None
        The cross-validation mean RMSE (k-fold) corresponding to the best-fit model.

    Methods
    -------
    numba_variogram(area_side, samples_per_area, max_samples, bin_width, max_lag_multiplier)
        Calculate the variogram using Numba for performance optimization.

    calculate_mean_variogram_numba(area_side, samples_per_area, max_samples, bin_width,
                                   max_n_bins, n_runs, max_lag_multiplier=1/3)
        Calculate the mean variogram by running the variogram calculation multiple times.

    fit_best_spherical_model()
        Systematically tests 1-, 2-, or 3-component spherical models (with/without nugget),
        chooses the best model by AIC, then runs a bootstrap to obtain parameter uncertainties.

    plot_best_spherical_model()
        Generates a two-panel plot showing a histogram of semivariance counts on top, and
        the mean variogram with the best-fit model and uncertainties on the bottom.
    """
    def __init__(self, raster_data_handler):
        """
        Initializes the VariogramAnalysis class with a RasterDataHandler instance.

        Parameters:
        -----------
        raster_data_handler : RasterDataHandler
            The RasterDataHandler instance to manage raster data operations.
        """
        self.raster_data_handler = raster_data_handler
        self.mean_variogram = None
        self.lags = None
        self.mean_count = None
        self.err_variogram = None
        self.best_model_config = None
        self.fitted_variogram = None
        self.rmse = None
        self.sills = None
        self.ranges = None
        self.ranges_min = None
        self.ranges_max = None
        self.ranges_median = None
        self.err_sills = None
        self.err_ranges = None
        self.sills_min = None
        self.sills_max = None
        self.sills_median = None
        self.best_nugget = None
        self.min_nugget = None
        self.max_nugget = None
        self.median_nugget = None
        self.best_aic = None
        self.best_params = None
        self.best_model_config = None
        self.cv_mean_error_best_aic = None
        self.param_samples = None
        self.n_bins = None
        self.sigma_variogram = None
        self.best_model_func = None
        self.best_guess = None
        self.best_bounds = None
        self.all_variograms = None
        self.all_counts = None
        self.param_samples = None
            
    @staticmethod
    @njit(parallel=True)
    def bin_distances_and_squared_differences(coords, values, bin_width, max_lag_multiplier, x_extent, y_extent):
        """
        Compute and bin pairwise distances and squared differences for Matheron estimation.

        Parameters:
        -----------
        coords : np.ndarray
            Array of coordinates of shape (M, 2).
        values : np.ndarray
            Array of values of shape (M,).
        bin_edges : np.ndarray
            Array of bin edges for distance binning.

        Returns:
        --------
        bin_counts : np.ndarray
            Counts of pairs in each bin.
        binned_sum_squared_diff : np.ndarray
            Sum of squared differences for each bin.
        """
        approx_max_distance = np.sqrt(x_extent**2 + y_extent**2)
        
        if max_lag_multiplier == "max":
            max_lag = approx_max_distance
        else:
            max_lag = int(approx_max_distance*max_lag_multiplier)
        
        #Determine bin edges using diagonal distance as maximum possible lag distance
        n_bins = int(np.ceil(max_lag / bin_width)) + 1
        bin_edges = np.arange(0, n_bins * bin_width, bin_width)
        
        M = coords.shape[0]
        max_distance = 0.0
        bin_counts = np.zeros(n_bins, dtype=np.int64)
        binned_sum_squared_diff = np.zeros(n_bins, dtype=np.float64)

        for i in prange(M):
            for j in range(i + 1, M):
                # Compute the pairwise distance
                d = 0.0
                for k in range(coords.shape[1]):
                    tmp = coords[i, k] - coords[j, k]
                    d += tmp * tmp
                dist = np.sqrt(d)
                max_distance = max(max_distance, dist)
                
                #Compute the difference
                diff = values[i] - values[j]
                
                # Compute the squared difference
                diff_squared = (diff) ** 2

                # Find the bin for this distance
                bin_idx = int(dist / bin_width)
                if 0 <= bin_idx < n_bins:
                    bin_counts[bin_idx] += 1
                    binned_sum_squared_diff[bin_idx] += diff_squared
        
        
        bin_edges = bin_edges[:n_bins]
        bin_counts = bin_counts[:n_bins]
        binned_sum_squared_diff = binned_sum_squared_diff[:n_bins]

        return n_bins, bin_counts, binned_sum_squared_diff, max_distance, max_lag
    
    @staticmethod
    def compute_matheron(bin_counts, binned_sum_squared_diff):
        """
        Compute the Matheron estimator from bin counts and squared differences.

        Parameters:
        -----------
        bin_counts : np.ndarray
            Counts of pairs in each bin.
        binned_sum_squared_diff : np.ndarray
            Sum of squared differences for each bin.

        Returns:
        --------
        matheron_estimates : np.ndarray
            Matheron variogram estimates for each bin.
        """
        matheron_estimates = np.zeros_like(bin_counts, dtype=np.float64)
        for i in range(len(bin_counts)):
            if bin_counts[i] > 0:
                matheron_estimates[i] = binned_sum_squared_diff[i] / (2 * bin_counts[i])
            else:
                matheron_estimates[i] = np.nan  # Handle empty bins
        return matheron_estimates
    
    def numba_variogram(self, area_side, samples_per_area, max_samples, bin_width, max_lag_multiplier):
        """
        Calculate the variogram using Numba for performance optimization.
        Parameters:
        -----------
        area_side : float
            The side length of the area to sample.
        samples_per_area : int
            The number of samples to take per area.
        max_samples : int
            The maximum number of samples to take.
        bin_width : float
            The width of the bins for distance binning.
        cell_size : float
            The size of the cell for declustering.
        n_offsets : int
            The number of offsets for declustering.
        max_lag_multiplier : str or float
            The multiplier for the maximum lag distance. Can be "median", "max", or a float value.
        normal_transform : bool
            Whether to apply a normal transformation to the data.
        weights : bool
            Whether to apply declustering weights.
        Returns:
        --------
        bin_counts : numpy.ndarray
            The counts of pairs in each bin.
        variogram_matheron : numpy.ndarray
            The calculated variogram values for each bin.
        n_bins : int
            The number of bins used.
        min_distance : float
            The minimum distance considered.
        max_distance : float
            The maximum distance considered.
        """
        
        self.raster_data_handler.sample_raster(area_side, samples_per_area, max_samples)
        
        min_distance = 0.0
        
        x_extent = self.raster_data_handler.rioxarray_obj.rio.width
        y_extent = self.raster_data_handler.rioxarray_obj.rio.height

        n_bins, bin_counts, binned_sum_squared_diff, max_distance, max_lag = self.bin_distances_and_squared_differences(self.raster_data_handler.coords, self.raster_data_handler.samples, bin_width, max_lag_multiplier, x_extent, y_extent)
        matheron_estimates = self.compute_matheron(bin_counts, binned_sum_squared_diff)
        
        return bin_counts, matheron_estimates, n_bins, min_distance, max_distance, max_lag
    
    def calculate_mean_variogram_numba(self, area_side, samples_per_area, max_samples, bin_width, max_n_bins, n_runs, max_lag_multiplier=1/3):
        
        """
        Calculate the mean variogram using numba for multiple runs.
        Parameters:
        -----------
        area_side : float
            The side length of the area to be sampled.
        samples_per_area : int
            The number of samples to be taken per area.
        max_samples : int
            The maximum number of samples to be taken.
        bin_width : float
            The width of each bin for the variogram.
        max_n_bins : int
            The maximum number of bins for the variogram.
        n_runs : int
            The number of runs to perform for averaging.
        max_lag_multiplier : float, optional
            The maximum lag distance as a fraction of the area side length (default is 1/3).

        Returns:
        --------
        None
        Attributes:
        -----------
        mean_variogram : numpy.ndarray
            The mean variogram calculated across all runs.
        err_variogram : numpy.ndarray
            The standard deviation of the variogram across all runs.
        mean_count : numpy.ndarray
            The mean count of pairs in each bin across all runs.
        lags : numpy.ndarray
            The lag distances corresponding to the variogram bins.
        """
        
        # Initialize DataFrames and arrays
        all_variograms = pd.DataFrame(np.nan, index=range(n_runs), columns=range(max_n_bins))
        counts = pd.DataFrame(np.nan, index=range(n_runs), columns=range(max_n_bins))
        

        # immediately store them on self so that fit_best_spherical_model can see them
        self.all_variograms = all_variograms
        self.all_counts     = counts
    
        # Initialize arrays to store bin information
        all_n_bins = np.zeros(n_runs)
        min_distances = np.zeros(n_runs)
        max_distances = np.zeros(n_runs)
        
        for run in range(n_runs):
            # Calculate variogram for all runs
            count, variogram, n_bins, min_distance, max_distance, max_lag = self.numba_variogram(area_side, samples_per_area, max_samples, bin_width, max_lag_multiplier)
            
            # Store the results
            self.all_variograms.loc[run, :variogram.size-1] = pd.Series(variogram).loc[:variogram.size-1]
            self.all_counts.loc[run, :count.size-1] = pd.Series(count).loc[:count.size-1]
            all_n_bins[run] = n_bins
            min_distances[run] = min_distance
            max_distances[run] = max_lag
        
        # 1) determine which columns (bins) ever contained data
        valid = (self.all_counts.sum(axis=0) > 0).values  # Boolean array, length = max_n_bins

        # 2) compute means over the valid bins only
        mean_variogram = self.all_variograms.loc[:, valid].mean(axis=0).values
        mean_count     = self.all_counts    .loc[:, valid].mean(axis=0).values

        # 3) compute std‐dev and 95% CI half‐width over the same bins
        sigma_variogram = self.all_variograms.loc[:, valid].std(axis=0).values
        sigma_variogram[sigma_variogram == 0] = np.finfo(float).eps

        ci_lower = self.all_variograms.loc[:, valid].quantile(0.025, axis=0).values
        ci_upper = self.all_variograms.loc[:, valid].quantile(0.975, axis=0).values
        err_variogram = (ci_upper - ci_lower) / 2

        # 4) build a lags array of the exact same length
        n_bins = len(mean_variogram)
        lags = np.linspace(bin_width / 2,
                        bin_width * n_bins - bin_width / 2,
                        n_bins)

        # 5) assign back to self
        self.mean_variogram = mean_variogram
        self.mean_count     = mean_count
        self.err_variogram  = err_variogram
        self.sigma_variogram= sigma_variogram
        self.lags           = lags
        self.n_bins         = n_bins
    
    @staticmethod
    def get_base_initial_guess(n, mean_variogram, lags, nugget=False):
        max_semivariance = np.max(mean_variogram)*1.5
        half_max_lag = np.max(lags) / 2
        C = [max_semivariance / n]*n      # sills
        a = [((half_max_lag)/3)*(i+1) for i in range(n)]  # ranges
        p0 = C + a + ([max_semivariance / 4] if nugget else [])
        return np.array(p0)
    
    @staticmethod
    def get_randomized_guesses(base_guess, n_starts=5, perturb_factor=0.3):
        """
        Generate multiple initial guesses by perturbing the base guess.
        'perturb_factor' is a fraction of the base guess values.
        """
        p0s = []
        for _ in range(n_starts):
            rand_perturbation = (np.random.rand(len(base_guess)) - 0.5) * 2.0
            # Scale by base_guess * perturb_factor
            new_guess = base_guess * (1 + rand_perturbation * perturb_factor)
            # Ensure no negative sills or other invalid guesses
            new_guess = np.clip(new_guess, 1e-3, None)  
            p0s.append(new_guess)
        return p0s
    
    @staticmethod
    def pure_nugget_model(h, nugget):
        return np.full_like(h, nugget)
    
    @staticmethod
    def spherical_model(h, *params):
        """
        Computes the spherical model for given distances and parameters.

        The spherical model is commonly used in geostatistics to describe spatial 
        correlation. It is defined piecewise, with different formulas for distances 
        less than or equal to the range parameter and for distances greater than the 
        range parameter.

        Parameters:
        h (array-like): Array of distances at which to evaluate the model.
        *params: Variable length argument list containing the sill and range parameters.
                    The first half of the parameters are the sills (C), and the second half 
                    are the ranges (a). The number of sills and ranges should be equal.

        Returns:
        numpy.ndarray: Array of model values corresponding to the input distances.
        """
        n = len(params) // 2
        C = params[:n]
        a = params[n:]
        model = np.zeros_like(h)
        for i in range(n):
            mask = h <= a[i]
            model[mask] += C[i] * (1.5 * (h[mask] / a[i]) - 0.5 * (h[mask] / a[i]) ** 3)
            #model[mask] += C[i] * (3 * h[mask] / (2 * a[i]) - (h[mask] ** 3) / (2 * a[i] ** 3))
            model[~mask] += C[i]
        return model
        
    def spherical_model_with_nugget(self,h, nugget, *params):
        """
        Computes the spherical model with a nugget effect.

        The spherical model is a type of variogram model used in geostatistics.
        This function adds a nugget effect to the spherical model.

        Parameters:
        h (array-like): Array of distances at which to evaluate the model.
        nugget (float): The nugget effect, representing the discontinuity at the origin.
        *params: Variable length argument list containing the sill and range parameters.
                    The first half of the parameters are the sills (C), and the second half 
                    are the ranges (a). The number of sills and ranges should be equal.

        Returns:
        numpy.ndarray: Array of model values corresponding to the input distances.
        """
        return nugget + self.spherical_model(h, *params)
    
    def cross_validate_variogram(self, model_func, p0, bounds, k=5):
        """
        Perform k-fold cross-validation on the binned variogram data.
        
        Returns a dictionary of average fold metrics:
            - 'rmse'
            - 'mae'
            - 'me'  (mean error)
            - 'mse'
        
        If no fold converges, returns None.
        """
        lags = self.lags
        mean_variogram = self.mean_variogram
        sigma_filtered = self.sigma_variogram
        
        n_bins = len(lags)
        indices = np.arange(n_bins)
        np.random.shuffle(indices)

        fold_size = max(1, n_bins // k)  # in case n_bins < k
        rmses = []
        maes = []
        mes  = []
        mses = []

        for i in range(k):
            valid_idx = indices[i*fold_size : (i+1)*fold_size]
            train_idx = np.setdiff1d(indices, valid_idx)

            # Subset train
            lags_train = lags[train_idx]
            vario_train = mean_variogram[train_idx]
            sigma_train = sigma_filtered[train_idx]

            # Fit on training fold
            try:
                popt, _ = curve_fit(model_func,
                                    lags_train, vario_train,
                                    p0=p0,
                                    bounds=bounds,
                                    sigma=sigma_train)
            except RuntimeError:
                continue

            # Predict on validation fold
            lags_valid = lags[valid_idx]
            vario_valid = mean_variogram[valid_idx]
            predictions = model_func(lags_valid, *popt)

            # Compute metrics
            errors = vario_valid - predictions
            rmse  = np.sqrt(np.mean(errors**2))
            mae   = np.mean(np.abs(errors))
            me    = np.mean(errors)
            mse   = np.mean(errors**2)

            rmses.append(rmse)
            maes.append(mae)
            mes.append(me)
            mses.append(mse)

        if len(rmses) == 0:
            return None  # indicates all folds failed to converge

        # Average across folds
        return {
            'rmse': np.mean(rmses),
            'mae':  np.mean(maes),
            'me':   np.mean(mes),
            'mse':  np.mean(mses)
        }
    
    @staticmethod
    def bootstrap_fit_variogram(lags, mean_vario,sigma_vario, model_func, p0,
                            bounds=None, n_boot=100, maxfev=10000):
        """
        Perform a parametric bootstrap to estimate parameter uncertainties for
        a variogram model, assuming `err_vario` represents the half-width of 
        a 95% confidence interval at each lag.

        Parameters
        ----------
        lags : np.ndarray
            Array of lag distances (length m).
        mean_vario : np.ndarray
            The mean variogram values across bins (length m).
        err_vario : np.ndarray 
            95% confidence interval for each bin (length m).
        model_func : callable
            The variogram model (e.g., spherical, exponential, etc.).
        p0 : np.ndarray
            An initial guess for the parameters [C1, C2, ..., a1, a2, ..., (nugget?)].
        bounds : tuple or None
            (lower_bounds, upper_bounds) for each parameter, if needed by curve_fit.
        n_boot : int
            Number of bootstrap replicates.
        maxfev : int
            Maximum function evaluations for curve_fit.

        Returns
        -------
        param_samples : np.ndarray
            Array of shape (n_successful, n_params) with fitted parameters from
            each bootstrap iteration. Some may fail to converge.
        """
        
        noise_array = np.zeros((n_boot, len(mean_vario)))
        rng = np.random.default_rng()
        for i,(v,s) in enumerate(zip(mean_vario,sigma_vario)):

            noise_temp = rng.normal(loc=v, scale=s, size=n_boot)
            noise_array[:,i] = noise_temp

        param_samples = []
        n_params = len(p0)

        for n in range(n_boot):

            # Create synthetic data
            synthetic_data = noise_array[n,:]

            # Fit the model
            try:
                popt_synth, pcov_synth = curve_fit(
                    model_func,
                    lags,
                    synthetic_data,
                    p0=p0,
                    sigma=None,  # or pass std_est if you want weighting
                    bounds=bounds if bounds is not None else (-np.inf, np.inf),
                    maxfev=maxfev
                )
                param_samples.append(popt_synth)
            except RuntimeError:
                # If the fit fails, store NaNs for the parameters
                param_samples.append([np.nan]*n_params)

        param_samples = np.array(param_samples)
        # Remove any failed fits (NaNs)
        valid = ~np.isnan(param_samples).any(axis=1)
        param_samples = param_samples[valid]

        return param_samples
        
    def fit_best_spherical_model(self):
        """
        Fits the best spherical model to the variogram data, potentially with a nugget term.
        This method systematically tries 1-, 2-, or 3-component spherical models, each with
        and without a nugget effect, selecting the best based on the Akaike Information
        Criterion (AIC). It then performs a parametric bootstrap to estimate uncertainties
        for the fitted parameters.

        Steps:
        1. Use `curve_fit` to fit each candidate model configuration.
        2. Compute AIC and keep track of the best solution.
        3. Once the best model is identified, perform a bootstrap to obtain
            percentile-based confidence intervals (min, max, median).
        4. Store all results in the instance attributes.

        Attributes Updated
        -----------------
        best_aic : float
            The lowest AIC value across all tried configurations.

        best_params : np.ndarray
            Fitted parameter array (sills, ranges, and possibly nugget) for the best model.

        best_model_config : dict
            Dictionary with keys 'components' and 'nugget', describing the best-fitting model.

        best_nugget : float or None
            Nugget parameter of the best model, if applicable.

        fitted_variogram : np.ndarray
            Semivariogram values computed from the best-fit model across the `self.lags` array.

        ranges, sills : np.ndarray or None
            Range(s) and sill(s) from the best-fitting model, each of length `n` if
            `components = n`, or None for a pure nugget.

        ranges_min, ranges_max, ranges_median : list or None
            2.5th percentile, 97.5th percentile, and median for each range parameter from
            bootstrapping, or None if pure nugget.

        sills_min, sills_max, sills_median : list or None
            2.5th percentile, 97.5th percentile, and median for each sill parameter.

        min_nugget, max_nugget, median_nugget : float or None
            2.5th percentile, 97.5th percentile, and median for the nugget parameter
            (only if a nugget is used).

        param_samples : np.ndarray
            The bootstrapped parameter vectors (shape: (n_successful, n_params)).

        cv_mean_error_best_aic : float
            Mean cross-validation error (RMSE) of the best-fitting model.

        Raises
        ------
        RuntimeError
            If no valid fit is found for any model or if bootstrapping yields no successful fits.

        Notes
        -----
        - Make sure `mean_variogram`, `err_variogram`, and `lags` are already computed
        via `calculate_mean_variogram_numba`.
        - The code uses random initial guesses per model configuration and discards fits
        that fail to converge.
        - For each final best-fit model, a parametric bootstrap is performed to estimate
        the confidence intervals of the sills, ranges, and nugget (if present).
    """
        
        # Initialize variables
        lags = self.lags
        mean_variogram = self.mean_variogram
        sigma_filtered = self.sigma_variogram

        # Define bounds and initial guesses for 1, 2, and 3 models
        model_configs = [
            #{'components': 0, 'nugget': True},
            {'components': 1, 'nugget': False},
            {'components': 1, 'nugget': True},
            {'components': 2, 'nugget': False},
            {'components': 2, 'nugget': True},
            {'components': 3, 'nugget': False},
            {'components': 3, 'nugget': True},
        ]
        best_local_aic = np.inf
        best_params = None
        best_model = None
        best_nugget = None

        for config in model_configs:
            n = config['components']
            nugget = config['nugget']
            
            if n==0:
                model_func = self.pure_nugget_model
                lower_bounds = [0]
                upper_bounds = [np.inf]
                bounds = (lower_bounds, upper_bounds)
                base_guess = [np.max(mean_variogram)]
                all_guesses = self.get_randomized_guesses(base_guess, n_starts=10, perturb_factor=0.5)
            else:
                lower_bounds = [0] * n + [1]* n + int(nugget)*[0]
                upper_bounds = [np.inf] * n + [np.max(lags)] * n + ([np.inf] if nugget else [])
                
                bounds = (lower_bounds, upper_bounds)
                
                model_func = self.spherical_model_with_nugget if nugget else self.spherical_model
                
                # Get a 'base guess'
                base_guess = self.get_base_initial_guess(n, mean_variogram, lags, nugget=nugget)
                
                # Generate multiple randomized guesses
                all_guesses = self.get_randomized_guesses(base_guess, n_starts=5, perturb_factor=0.5)
            
            for guess in all_guesses:
                try:
                    popt, pcov = curve_fit(model_func, lags, mean_variogram,
                                    p0=guess, bounds=bounds,
                                    sigma=sigma_filtered, maxfev=10000)
                    
                    param_samples = []
                    
                    # Number of bins in the mean variogram
                    n_bins_mean = len(self.mean_variogram)

                    for i in range(len(self.all_variograms.index)):
                        # Extract the i-th run for exactly those mean-variogram bins
                        row = self.all_variograms.iloc[i, :n_bins_mean]
                        # Boolean mask of non-NaN entries in this run
                        mask = ~row.isna().values

                        # Align lags and semivariances for fitting
                        lags_i = lags[mask]
                        semivariance_values = row.values[mask]

                        try:
                            popt_synth, pcov_synth = curve_fit(
                                model_func,
                                lags_i,
                                semivariance_values,
                                p0=guess,
                                sigma=None,
                                bounds=bounds,
                                maxfev=10000
                            )
                            param_samples.append(popt_synth)
                        except RuntimeError:
                            # preserve shape for downstream percentile calculations
                            param_samples.append([np.nan] * len(guess))
        
        
                    # # Precompute which bins are valid in the mean variogram
                    # valid_bins = ~np.isnan(self.mean_variogram)
                    # lags_valid = lags[valid_bins]

                    # for i in range(len(self.all_variograms.index)):
                    #     # pull only the semivariance values for those same valid bins
                    #     semivariance_values = self.all_variograms.iloc[i, valid_bins].values
                    #     try:
                    #         popt_synth, pcov_synth = curve_fit(
                    #             model_func,
                    #             lags_valid,
                    #             semivariance_values,
                    #             p0=guess,
                    #             sigma=None,
                    #             bounds=bounds,
                    #             maxfev=10000
                    #         )
                    #         param_samples.append(popt_synth)
                    #     except RuntimeError:
                    #         # keep the same shape so downstream percentiles work
                    #         param_samples.append([np.nan] * len(guess))
                    
                    # for i in range (0,len(self.all_variograms.index)):
                    #     semivariance_values = self.all_variograms.iloc[i,:].dropna().values
                    #     try:
                    #         popt_synth, pcov_synth = curve_fit(
                    #             model_func,
                    #             lags,
                    #             semivariance_values,
                    #             p0=guess,
                    #             sigma=None,  # or pass std_est if you want weighting
                    #             bounds=bounds if bounds is not None else (-np.inf, np.inf),
                    #             maxfev=10000
                    #         )
                    #         param_samples.append(popt_synth)
                    #     except RuntimeError:
                    #         # If the fit fails, store NaNs for the parameters
                    #         param_samples.append([np.nan]*n)

                    param_samples = np.array(param_samples)
                    # Remove any failed fits (NaNs)
                    valid = ~np.isnan(param_samples).any(axis=1)
                    param_samples = param_samples[valid]
                    
                    if len(popt) == 0:
                        range_mins = range_maxs = range_medians = None
                        sill_mins = sill_maxs = sill_medians = None
                        nugget_min = nugget_max = nugget_median = None
                        raise RuntimeError("No successful fits in bootstrap")
                            
                    if config['nugget'] and n == 0:
                        sills = None
                        ranges = None
                        nugget_value = popt[0]
                        
                        nugget_samples = [a[0] for a in param_samples]
                        nugget_samples = np.array(nugget_samples)
                        nugget_min = np.percentile(nugget_samples, 16)
                        nugget_max = np.percentile(nugget_samples, 84)
                        nugget_median = np.percentile(nugget_samples, 50)
                        
                    elif config['nugget'] and n > 0:
                        sills = popt[0:n]
                        ranges = popt[n:n+n]
                        nugget_value = popt[-1]
                         
                        sill_samples = [a[0:n] for a in param_samples]
                        sill_samples = np.array(sill_samples)
                        range_samples = [a[n:n+n] for a in param_samples]
                        range_samples = np.array(range_samples)
                        nugget_samples = [a[-1] for a in param_samples]
                        nugget_samples = np.array(nugget_samples)
                        
                        sill_mins = [np.percentile(sill_samples[:,i], 16) for i in range(n)]
                        sill_maxs = [np.percentile(sill_samples[:,i], 84) for i in range(n)]
                        sill_medians = [np.percentile(sill_samples[:,i], 50) for i in range(n)]
                        
                        range_mins = [np.percentile(range_samples[:,i], 16) for i in range(n)]
                        range_maxs = [np.percentile(range_samples[:,i], 84) for i in range(n)]
                        range_medians = [np.percentile(range_samples[:,i], 50) for i in range(n)]
                        
                        err_ranges = np.array([(a - b)/2 for a, b in zip(range_maxs, range_mins)])
                        err_sills = np.array([(a - b)/2 for a, b in zip(sill_maxs, sill_mins)])

                        prop_range = np.array([a / b for a, b in zip(err_ranges, ranges)])
                        prop_sill = np.array([a / b for a, b in zip(err_sills, sills)])
                        
                        nugget_min = np.percentile(nugget_samples, 2.5)
                        nugget_max = np.percentile(nugget_samples, 97.5)
                        nugget_median = np.percentile(nugget_samples, 50)
                        
                    else:
                        sills = popt[0:n]
                        ranges = popt[n:n+n] 
                        nugget_value = None
                        
                        sill_samples = [a[0:n] for a in param_samples]
                        sill_samples = np.array(sill_samples)
                        range_samples = [a[n:n+n] for a in param_samples]
                        range_samples = np.array(range_samples)
                        nugget_samples = None
                        
                        sill_mins = [np.percentile(sill_samples[:,i], 16) for i in range(n)]
                        sill_maxs = [np.percentile(sill_samples[:,i], 84) for i in range(n)]
                        sill_medians = [np.percentile(sill_samples[:,i], 50) for i in range(n)]
                        
                        range_mins = [np.percentile(range_samples[:,i], 16) for i in range(n)]
                        range_maxs = [np.percentile(range_samples[:,i], 84) for i in range(n)]
                        range_medians = [np.percentile(range_samples[:,i], 50) for i in range(n)]
                    
                        err_ranges = np.array([(a - b)/2 for a, b in zip(range_maxs, range_mins)])
                        err_sills = np.array([(a - b)/2 for a, b in zip(sill_maxs, sill_mins)])

                        prop_range = np.array([a / b for a, b in zip(err_ranges, ranges)])
                        prop_sill = np.array([a / b for a, b in zip(err_sills, sills)])
                        
                        nugget_min = None
                        nugget_max = None
                        nugget_median = None
                    
                    fitted_variogram = model_func(self.lags, *popt)
                    residuals = self.mean_variogram - fitted_variogram
                    n_data_points = len(self.mean_variogram)
                    residual_sum_of_squares = np.sum(residuals**2)
                    sigma_squared = np.var(residuals)
                    if sigma_squared <= 0: # Avoid taking the log of zero or negative variance
                        sigma_squared = np.finfo(float).eps  # Use a small value instead
                    log_likelihood = -0.5 * n_data_points * np.log(2 * np.pi * sigma_squared) - residual_sum_of_squares / (2 * sigma_squared)
                    k = len(popt)
                    aic = 2 * k - 2 * log_likelihood
                    
                    rmse_filtered = np.sqrt(np.mean((mean_variogram - model_func(lags, *popt))**2))
                    rmse = np.sqrt(np.mean((self.mean_variogram - fitted_variogram)**2))

                    if (
                        aic < best_local_aic
                        and (
                            ranges is None
                            or (
                                all(r > 0 for r in range_mins)
                                and all(r < 0.75*np.max(lags) for r in range_maxs)
                            )
                        )
                    #    and (
                    #        all(r < 0.8 for r in prop_range)
                    #        and all(r < 0.8 for r in prop_sill)
                    #    )
                    ):
                        best_local_aic = aic
                        best_params = popt
                        best_model = config
                        best_func = model_func
                        best_nugget = nugget_value
                        best_ranges = ranges
                        best_sills = sills
                        best_range_mins = range_mins
                        best_range_maxs = range_maxs
                        best_range_medians = range_medians
                        best_sill_mins = sill_mins
                        best_sill_maxs = sill_maxs
                        best_sill_medians = sill_medians
                        best_nugget_min = nugget_min
                        best_nugget_max = nugget_max
                        best_nugget_median = nugget_median
                        best_fitted = fitted_variogram
                        best_err_sills = err_sills
                        best_err_ranges = err_ranges
                        best_guess = guess
                        best_bounds = bounds
                    
                except RuntimeError:
                    continue
        
        if best_model is not None:
            
            self.best_aic = best_local_aic
            self.best_params = best_params
            self.ranges = best_ranges
            self.sills = best_sills
            self.best_nugget = best_nugget
            self.fitted_variogram = best_fitted
            self.best_model_config = best_model
            self.best_model_func = best_func
            self.best_guess = best_guess
            self.best_bounds = best_bounds
            self.param_samples = param_samples
            
            self.ranges_min = best_range_mins
            self.ranges_max = best_range_maxs
            self.ranges_median = best_range_medians
            self.sills_min = best_sill_mins
            self.sills_max = best_sill_maxs
            self.sills_median = best_sill_medians
            self.min_nugget = best_nugget_min
            self.max_nugget = best_nugget_max
            self.median_nugget = best_nugget_median
            self.err_sills = best_err_sills
            self.err_ranges = best_err_ranges
            
                  
    def plot_best_spherical_model(self):
        """
        Plots the best spherical model for the variogram analysis.
        This function generates a two-panel plot:
        - The top panel displays a histogram of semivariance counts.
        - The bottom panel shows the mean variogram with error bars, the fitted model, range values with their errors, and the nugget effect if applicable.
        The plot includes:
        - A histogram of semivariance counts in the top panel.
        - The mean variogram with error bars indicating the spread over multiple runs.
        - The fitted variogram model if available.
        - Vertical lines indicating range values and shaded areas representing their errors.
        - A horizontal line indicating the nugget effect if used.
        - The RMSE value in the title of the bottom panel.
        Returns:
            fig (matplotlib.figure.Figure): The figure object containing the plot.
        """
            # Align all arrays to the same base length
        n = len(self.lags)
        lags_full   = self.lags
        gamma_full  = self.mean_variogram
        errs_full   = self.err_variogram
        counts_full = self.mean_count
        model_full  = self.fitted_variogram

        # Truncate any longer arrays down to n
        errs_full   = errs_full[:n]
        counts_full = counts_full[:n]
        model_full  = model_full[:n]

        # Mask out any NaNs in the mean variogram
        mask = ~np.isnan(gamma_full)

        lags   = lags_full[mask]
        gamma  = gamma_full[mask]
        errs   = errs_full[mask]
        counts = counts_full[mask & (counts_full > 0)]
        model  = model_full[mask]
        # valid = ~np.isnan(self.mean_variogram)
        # lags   = self.lags[valid]
        # gamma  = self.mean_variogram[valid]
        # errs   = self.err_variogram[valid]
        # counts = self.mean_count[(~np.isnan(self.mean_count)) & (self.mean_count != 0)]
        # model = self.fitted_variogram[valid]
        
        fig, ax = plt.subplots(2, 1, gridspec_kw={'height_ratios': [1, 3]}, figsize=(10, 8))
        
        
         # Histogram of semivariance counts
        ax[0].bar(lags, counts, width=np.diff(lags)[0] * 0.9, color='orange', alpha=0.5)
        ax[0].set_ylabel('Mean Count')

        # Plot mean variogram with error bars indicating spread over the 10 runs
        ax[1].errorbar(lags, gamma, yerr=errs, fmt='o-', color='blue', label='Mean Variogram with Spread')

        # Plot fitted model
        if self.fitted_variogram is not None:
            ax[1].plot(lags, model, 'r-', label='Fitted Model')

        colors = ['red', 'green', 'blue']
        if self.ranges is not None and self.ranges_min is not None and self.ranges_max is not None:
            for i, (range_val, min_range_val, max_range_val) in enumerate(zip(self.ranges, self.ranges_min, self.ranges_max)):
                color = colors[i]
                # Bold line at range value
                ax[1].axvline(x=range_val, color=color, linestyle='--', lw=1, label=f'Range {i+1}')
                # Lighter lines at +/- 1 std error
                ax[1].fill_betweenx([0, np.max(self.mean_variogram)], min_range_val, max_range_val, color=color, alpha=0.2)

       
        # Check if a nugget effect was used
        if self.best_nugget is not None and self.min_nugget is not None and self.max_nugget is not None:
            ax[1].axhline(y=self.best_nugget, color='black', linestyle='--', lw=1, label='Nugget Effect')
            ax[1].fill_between([0, np.max(self.lags)], self.min_nugget, self.max_nugget, color='gray', alpha=0.2)

        ax[1].set_xlabel('Lag Distance')
        ax[1].set_ylabel('Semivariance')
        ax[1].legend()

        # Show both AIC and RMSE in the title
       #aic_str = f'AIC: {self.best_aic:.4f}' if self.best_aic is not None else "AIC: N/A"
        if self.cv_mean_error_best_aic is not None:
        # Assuming the dictionary has a key 'rmse'
            rmse_value = self.cv_mean_error_best_aic.get('rmse')
            if rmse_value is not None:
                rmse_str = f'RMSE: {rmse_value:.4f}' 
            else:
                rmse_str = "RMSE: N/A"
            ax[1].set_title(f'{rmse_str}')
        #ax[1].set_title(f'{aic_str}, {rmse_str}')
        

        plt.setp(ax[0].get_xticklabels(), visible=False)
        plt.tight_layout()

        return fig

class UncertaintyCalculation:
    """
    A class designed to calculate various types of uncertainty associated with spatial data,
    particularly focusing on the uncertainty derived from variogram analysis.

    Attributes:
    -----------
    variogram_analysis : VariogramAnalysis
        An instance of VariogramAnalysis containing variogram results and fitted model parameters.
    mean_random_uncorrelated : float
        The mean random uncertainty not correlated to any spatial structure.
    mean_random_correlated_1 : float
        The mean random uncertainty correlated to the first spherical model.
    mean_random_correlated_2 : float
        The mean random uncertainty correlated to the second spherical model.
    mean_random_correlated_3 : float
        The mean random uncertainty correlated to the third spherical model.
    mean_random_correlated_1_min : float
        The minimum mean random uncertainty correlated to the first spherical model.
    mean_random_correlated_2_min : float
        The minimum mean random uncertainty correlated to the second spherical model.
    mean_random_correlated_3_min : float
        The minimum mean random uncertainty correlated to the third spherical model.
    mean_random_correlated_1_max : float
        The maximum mean random uncertainty correlated to the first spherical model.
    mean_random_correlated_2_max : float
        The maximum mean random uncertainty correlated to the second spherical model.
    mean_random_correlated_3_max : float
        The maximum mean random uncertainty correlated to the third spherical model.
    total_mean_uncertainty : float
        The total mean uncertainty calculated as a combination of uncorrelated and correlated uncertainties.
    total_mean_uncertainty_min : float
        The minimum total mean uncertainty calculated as a combination of uncorrelated and correlated uncertainties.
    total_mean_uncertainty_max : float
        The maximum total mean uncertainty calculated as a combination of uncorrelated and correlated uncertainties.
    area : float
        The area associated with the uncertainty calculation.

    Methods:
    --------
    calc_mean_random_uncorrelated()
        Calculates the mean random uncorrelated uncertainty.
    calc_mean_random_correlated()
        Calculates the mean random correlated uncertainties for each spherical model component.
    calc_mean_random_correlated_min()
        Calculates the mean random correlated uncertainties for the minimum sills of the variogram analysis.
    calc_mean_random_correlated_max()
        Calculates the mean random correlated uncertainties for the maximum sills of the variogram analysis.
    calc_total_mean_uncertainty()
        Calculates the total mean uncertainty by adding in quadrature the uncertainties (both correlated and uncorrelated).
    calc_total_mean_uncertainty_min()
        Calculates the minimum total mean uncertainty by adding in quadrature the uncertainties (both minimum correlated and uncorrelated).
    calc_total_mean_uncertainty_max()
        Calculates the maximum total mean uncertainty by adding in quadrature the uncertainties (both maximum correlated and uncorrelated).
    """

    def __init__(self, variogram_analysis):
        """
        Initialize the UncertaintyCalculation class with a variogram analysis object.
        Parameters:
        variogram_analysis (object): An object containing variogram analysis data.
        Attributes:
        mean_random_uncorrelated (float or None): Mean random uncorrelated uncertainty.
        mean_random_correlated_1 (float or None): Mean random correlated uncertainty for the first correlation.
        mean_random_correlated_2 (float or None): Mean random correlated uncertainty for the second correlation.
        mean_random_correlated_3 (float or None): Mean random correlated uncertainty for the third correlation.
        mean_random_correlated_1_min (float or None): Minimum mean random correlated uncertainty for the first correlation.
        mean_random_correlated_2_min (float or None): Minimum mean random correlated uncertainty for the second correlation.
        mean_random_correlated_3_min (float or None): Minimum mean random correlated uncertainty for the third correlation.
        mean_random_correlated_1_max (float or None): Maximum mean random correlated uncertainty for the first correlation.
        mean_random_correlated_2_max (float or None): Maximum mean random correlated uncertainty for the second correlation.
        mean_random_correlated_3_max (float or None): Maximum mean random correlated uncertainty for the third correlation.
        total_mean_uncertainty (float or None): Total mean uncertainty.
        total_mean_uncertainty_min (float or None): Minimum total mean uncertainty.
        total_mean_uncertainty_max (float or None): Maximum total mean uncertainty.
        area (float or None): Area associated with the uncertainty calculation.
        """
        
        self.variogram_analysis = variogram_analysis
        # Initialize all uncertainty attributes to None.
        self.mean_random_uncorrelated = None
        self.mean_random_correlated_1 = None
        self.mean_random_correlated_2 = None
        self.mean_random_correlated_3 = None
        self.mean_random_correlated_1_min = None
        self.mean_random_correlated_2_min = None
        self.mean_random_correlated_3_min = None
        self.mean_random_correlated_1_max = None
        self.mean_random_correlated_2_max = None
        self.mean_random_correlated_3_max = None
        self.total_mean_uncertainty = None
        self.total_mean_uncertainty_min = None
        self.total_mean_uncertainty_max = None
        self.area=None
    
    

    def calc_mean_random_uncorrelated(self):
        """
        Calculate the mean random uncorrelated uncertainty.
        
        4. Compute the mean random uncorrelated uncertainty by dividing the RMS by the square root of the length of the data array.
        The result is stored in the `mean_random_uncorrelated` attribute of the instance.
        """
        
        data = self.variogram_analysis.raster_data_handler.data_array

        def calculate_rms(values):
            """Calculate the Root Mean Square (RMS) of an array of numbers."""
            # Step 1: Square all the numbers
            squared_values = [x**2 for x in values]
            
            # Step 2: Calculate the mean of the squares
            mean_of_squares = sum(squared_values) / len(values)
            
            # Step 3: Take the square root of the mean
            rms = math.sqrt(mean_of_squares)
    
            return rms
        
        rms = calculate_rms(data)

        self.mean_random_uncorrelated = rms / np.sqrt(len(data))

    def calc_mean_random_correlated(self):
        """
        Calculate the mean random correlated uncertainties for each spherical model component.
        This method computes the correlated uncertainties based on the variogram analysis 
        parameters such as sills and ranges, and the resolution of the raster data. The 
        uncertainties are calculated for up to three spherical model components if available.
        Attributes:
        -----------
        mean_random_correlated_1 : float
            The mean random correlated uncertainty for the first spherical model component.
        mean_random_correlated_2 : float, optional
            The mean random correlated uncertainty for the second spherical model component, 
            if available.
        mean_random_correlated_3 : float, optional
            The mean random correlated uncertainty for the third spherical model component, 
            if available.
        area : float
            The total area covered by the raster data.
        """
        
        dem_resolution = self.variogram_analysis.raster_data_handler.resolution
        data = self.variogram_analysis.raster_data_handler.data_array
        # Calculate correlated uncertainties for each spherical model component.
        self.mean_random_correlated_1 = (np.sqrt(2 * self.variogram_analysis.sills[0]) / np.sqrt(len(data))) * np.sqrt((np.pi * np.square(self.variogram_analysis.ranges[0])) / (5 * np.square(dem_resolution)))
        if len(self.variogram_analysis.ranges)>1:
            self.mean_random_correlated_2 = (np.sqrt(2 * self.variogram_analysis.sills[1]) / np.sqrt(len(data))) * np.sqrt((np.pi * np.square(self.variogram_analysis.ranges[1])) / (5 * np.square(dem_resolution)))
            if len(self.variogram_analysis.ranges)>2:
                self.mean_random_correlated_3 = (np.sqrt(2 * self.variogram_analysis.sills[2]) / np.sqrt(len(data))) * np.sqrt((np.pi * np.square(self.variogram_analysis.ranges[2])) / (5 * np.square(dem_resolution)))    
        self.area=dem_resolution*len(data)

    def calc_mean_random_correlated_min(self):
        """
        Calculate the mean random correlated uncertainties for the minimum sills of the variogram analysis.
        This method calculates the mean random correlated uncertainties for each spherical model component
        based on the minimum sills and ranges from the variogram analysis. The calculation is performed
        only if the sill value is greater than zero. The results are stored in the attributes:
        `mean_random_correlated_1_min`, `mean_random_correlated_2_min`, and `mean_random_correlated_3_min`.
        Attributes:
            mean_random_correlated_1_min (float): Mean random correlated uncertainty for the first spherical model component.
            mean_random_correlated_2_min (float): Mean random correlated uncertainty for the second spherical model component.
            mean_random_correlated_3_min (float): Mean random correlated uncertainty for the third spherical model component.
        Returns:
            None
        """
        
        dem_resolution = self.variogram_analysis.raster_data_handler.resolution
        data = self.variogram_analysis.raster_data_handler.data_array
        # Calculate correlated uncertainties for each spherical model component.
        if self.variogram_analysis.sills_min[0]>0:
            self.mean_random_correlated_1_min = (np.sqrt(2 * self.variogram_analysis.sills_min[0]) / np.sqrt(len(data))) * np.sqrt((np.pi * np.square(self.variogram_analysis.ranges_min[0])) / (5 * np.square(dem_resolution)))
        else:
            self.mean_random_correlated_1_min = 0
        if len(self.variogram_analysis.ranges)>1:
            if self.variogram_analysis.sills_min[1]>0:
                self.mean_random_correlated_2_min = (np.sqrt(2 * self.variogram_analysis.sills_min[1]) / np.sqrt(len(data))) * np.sqrt((np.pi * np.square(self.variogram_analysis.ranges_min[1])) / (5 * np.square(dem_resolution)))
            else:
                self.mean_random_correlated_2_min = 0
            if len(self.variogram_analysis.ranges)>2:
                if self.variogram_analysis.sills_min[2]>0:
                    self.mean_random_correlated_3_min = (np.sqrt(2 * self.variogram_analysis.sills_min[2]) / np.sqrt(len(data))) * np.sqrt((np.pi * np.square(self.variogram_analysis.ranges_min[2])) / (5 * np.square(dem_resolution)))
                else:
                    self.mean_random_correlated_3_min = 0

    def calc_mean_random_correlated_max(self):
        """
        Calculate the mean random correlated uncertainties for each spherical model component.
        This method computes the mean random correlated uncertainties using the maximum sills and ranges
        from the variogram analysis. The calculation is performed for each spherical model component
        present in the variogram analysis.
        The formula used for the calculation is:
        mean_random_correlated = (sqrt(2 * sill) / sqrt(len(data))) * sqrt((pi * range^2) / (5 * dem_resolution^2))
        Attributes:
        -----------
        mean_random_correlated_1_max : float
            The mean random correlated uncertainty for the first spherical model component.
        mean_random_correlated_2_max : float, optional
            The mean random correlated uncertainty for the second spherical model component, if present.
        mean_random_correlated_3_max : float, optional
            The mean random correlated uncertainty for the third spherical model component, if present.
        Notes:
        ------
        - The method assumes that the variogram analysis has at least one spherical model component.
        - The method updates the instance attributes `mean_random_correlated_1_max`, `mean_random_correlated_2_max`,
          and `mean_random_correlated_3_max` based on the number of spherical model components.
        """
        
        dem_resolution = self.variogram_analysis.raster_data_handler.resolution
        data = self.variogram_analysis.raster_data_handler.data_array
        # Calculate correlated uncertainties for each spherical model component.
        self.mean_random_correlated_1_max = (np.sqrt(2 * self.variogram_analysis.sills_max[0]) / np.sqrt(len(data))) * np.sqrt((np.pi * np.square(self.variogram_analysis.ranges_max[0])) / (5 * np.square(dem_resolution)))
        if len(self.variogram_analysis.ranges)>1:
            self.mean_random_correlated_2_max = (np.sqrt(2 * self.variogram_analysis.sills_max[1]) / np.sqrt(len(data))) * np.sqrt((np.pi * np.square(self.variogram_analysis.ranges_max[1])) / (5 * np.square(dem_resolution)))
            if len(self.variogram_analysis.ranges)>2:
                self.mean_random_correlated_3_max = (np.sqrt(2 * self.variogram_analysis.sills_max[2]) / np.sqrt(len(data))) * np.sqrt((np.pi * np.square(self.variogram_analysis.ranges_max[2])) / (5 * np.square(dem_resolution)))    
             
    def calc_total_mean_uncertainty(self):
        """
        Calculates the total mean uncertainty by adding in quadrature the uncertainties (both correlated and uncorrelated).
        """
        if len(self.variogram_analysis.ranges) ==1:
            self.total_mean_uncertainty = np.sqrt(np.square(self.mean_random_uncorrelated) + np.square(self.mean_random_correlated_1))
        elif len(self.variogram_analysis.ranges) ==2:
            self.total_mean_uncertainty = np.sqrt(np.square(self.mean_random_uncorrelated) + np.square(self.mean_random_correlated_1) + np.square(self.mean_random_correlated_2))
        elif len(self.variogram_analysis.ranges) ==3:
            self.total_mean_uncertainty = np.sqrt(np.square(self.mean_random_uncorrelated) + np.square(self.mean_random_correlated_1) + np.square(self.mean_random_correlated_2) + np.square(self.mean_random_correlated_3))
    
    def calc_total_mean_uncertainty_min(self):
        """
        Calculates the minimum total mean uncertainty by adding in quadrature the uncertainties (both minimum correlated and uncorrelated).
        """
        if len(self.variogram_analysis.ranges) ==1:
            self.total_mean_uncertainty_min = np.sqrt(np.square(self.mean_random_uncorrelated) + np.square(self.mean_random_correlated_1_min))
        elif len(self.variogram_analysis.ranges) ==2:
            self.total_mean_uncertainty_min = np.sqrt(np.square(self.mean_random_uncorrelated) + np.square(self.mean_random_correlated_1_min) + np.square(self.mean_random_correlated_2_min))
        elif len(self.variogram_analysis.ranges) ==3:
            self.total_mean_uncertainty_min = np.sqrt(np.square(self.mean_random_uncorrelated) + np.square(self.mean_random_correlated_1_min) + np.square(self.mean_random_correlated_2_min) + np.square(self.mean_random_correlated_3_min))

    def calc_total_mean_uncertainty_max(self):
        """
        Calculates the maximum total mean uncertainty by adding in quadrature the uncertainties (both maximum correlated and uncorrelated).
        """
        if len(self.variogram_analysis.ranges) ==1:
            self.total_mean_uncertainty_max = np.sqrt(np.square(self.mean_random_uncorrelated) + np.square(self.mean_random_correlated_1_max))
        elif len(self.variogram_analysis.ranges) ==2:
            self.total_mean_uncertainty_max = np.sqrt(np.square(self.mean_random_uncorrelated) + np.square(self.mean_random_correlated_1_max) + np.square(self.mean_random_correlated_2_max))
        elif len(self.variogram_analysis.ranges) ==3:
            self.total_mean_uncertainty_max = np.sqrt(np.square(self.mean_random_uncorrelated) + np.square(self.mean_random_correlated_1_max) + np.square(self.mean_random_correlated_2_max) + np.square(self.mean_random_correlated_3_max))

class ApplyUncertainty:
    """
    Compute spatial (correlated + uncorrelated) uncertainties from variogram parameters,
    and compute RMS from a GeoTIFF band.
    """

    @staticmethod
    def compute_spatial_uncertainties(
        ranges: Sequence[float],
        sills: Sequence[float],
        area: float,
        resolution: float,
        rms: Optional[float] = None,
        sills_min: Optional[Sequence[float]] = None,
        ranges_min: Optional[Sequence[float]] = None,
        sills_max: Optional[Sequence[float]] = None,
        ranges_max: Optional[Sequence[float]] = None
    ) -> Dict[str, Any]:
        """
        Compute mean uncorrelated & correlated uncertainty terms and their quadrature sum.

        Parameters
        ----------
        ranges : sequence of float
            Range parameters from variogram (same units as resolution).
        sills : sequence of float
            Sill parameters from variogram (variance units).
        area : float
            Total sampling area (in same linear units squared as ranges).
        resolution : float
            Raster cell size (same linear units as ranges).
        rms : float, optional
            Root‐mean‐square of your data array. If supplied, used for uncorrelated term.
        sills_min, ranges_min, sills_max, ranges_max : sequences, optional
            Percentile bounds for sills/ranges to get total_min/total_max.

        Returns
        -------
        dict
            {
              'uncorrelated': float or None,
              'correlated': List[float],
              'total': float,
              'total_min': float or None,
              'total_max': float or None
            }
        """
        # effective sample count
        n = area / (resolution ** 2)

        # uncorrelated term (if rms given)
        uncorr = (rms / math.sqrt(n)) if rms is not None else None

        # correlated terms
        corr = []
        for sill, rng in zip(sills, ranges):
            term = (math.sqrt(2 * sill) / math.sqrt(n)) * \
                   math.sqrt((math.pi * rng**2) / (5 * resolution**2))
            corr.append(term)

        # total quadrature sum
        total_sq = sum(c**2 for c in corr) + ((uncorr**2) if uncorr is not None else 0)
        total = math.sqrt(total_sq)

        # optional bounds
        total_min = total_max = None
        if sills_min and ranges_min:
            corr_min = [
                (math.sqrt(2 * smin) / math.sqrt(n)) * math.sqrt((math.pi * rmin**2) / (5 * resolution**2))
                for smin, rmin in zip(sills_min, ranges_min)
            ]
            total_min = math.sqrt(sum(c**2 for c in corr_min) + ((uncorr**2) if uncorr is not None else 0))

        if sills_max and ranges_max:
            corr_max = [
                (math.sqrt(2 * smax) / math.sqrt(n)) * math.sqrt((math.pi * rmax**2) / (5 * resolution**2))
                for smax, rmax in zip(sills_max, ranges_max)
            ]
            total_max = math.sqrt(sum(c**2 for c in corr_max) + ((uncorr**2) if uncorr is not None else 0))

        return {
            'uncorrelated': uncorr,
            'correlated': corr,
            'total': total,
            'total_min': total_min,
            'total_max': total_max
        }

    @staticmethod
    def compute_rms_from_tif(
        tif_path: str,
        band: int = 1
    ) -> float:
        """
        Compute the root‐mean‐square of a GeoTIFF band, ignoring nodata and NaNs.

        Parameters
        ----------
        tif_path : str
            Path to the input GeoTIFF.
        band : int, default 1
            Raster band to read.

        Returns
        -------
        float
            RMS of all valid (non‐nodata, non‐NaN) pixels.
        """
        with rasterio.open(tif_path) as src:
            arr = src.read(band).astype(float)
            nodata = src.nodata

        valid = ~np.isnan(arr)
        if nodata is not None:
            valid &= (arr != nodata)

        vals = arr[valid]
        if vals.size == 0:
            raise ValueError("No valid pixels found (all nodata or NaN).")
        return float(np.sqrt(np.mean(vals**2)))