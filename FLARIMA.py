#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
from astropy.timeseries import LombScargle
from scipy.interpolate import UnivariateSpline
from lightkurve import TessLightCurveFile
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from tqdm import tqdm
from astroquery.mast import Observations, Tesscut
from astroquery.mast import Observations
import astropy.units as u
from pmdarima import auto_arima
import os
from astropy.table import Table
from astropy.io import fits
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from joblib import Parallel, delayed

def analyze_tess_data_from_directory(directory):
    lc_file_paths = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('_lc.fits')]
    
    for file_path in lc_file_paths:
        lcf = TessLightCurveFile(file_path)
        flux = lcf.flux
        flux_err = lcf.flux_err
        time = lcf.time.value
        time_days = time
        median_flux = np.median(flux)
        normalized_flux = flux / median_flux
        
# Detrend the data using cubic splines
        knots = np.linspace(time_days.min(), time_days.max(), 20)
        spl = UnivariateSpline(time_days, normalized_flux, k=3, s=0)
        detrended_flux = normalized_flux - spl(time_days)

# Reevaluate ARIMA Model Parameters
        p_values = range(0, 5)
        d_values = range(0, 3)
        q_values = range(0, 5)

# Hyperparameter Optimization
        total_iterations = len(p_values) * len(d_values) * len(q_values)
        progress_bar = tqdm(total=total_iterations, desc="Searching for best ARIMA model", position=0)

        best_aic = float("inf")
        best_model = None

        warnings.filterwarnings("ignore", category=ConvergenceWarning)  # Ignore ConvergenceWarning

        for p in p_values:
            for d in d_values:
                for q in q_values:
                    progress_bar.update(1)  # Update the progress bar
                    try:
                        model = ARIMA(normalized_flux, order=(p, d, q))
                        with warnings.catch_warnings():
                            warnings.filterwarnings("ignore")  # Ignore all warnings inside the context manager
                            model_fit = model.fit()
                        aic = model_fit.aic

                        if aic < best_aic:
                            best_aic = aic
                            best_model = model_fit
                    except:
                        continue

        progress_bar.close()

# Predict flux using the best-fit ARIMA model
        predicted_flux = best_model.predict(start=0, end=len(time_days) - 1)

# Calculate residuals
        residuals = normalized_flux - predicted_flux

# Detect flares
        flare_threshold = np.std(residuals) * 3
        flare_indices = np.where(residuals > flare_threshold)[0]
        flare_times = time_days[flare_indices]

# Plotting
        fig, axs = plt.subplots(2, 1, figsize=(12, 18), sharex=True, gridspec_kw={'hspace': 0})

        # Plot 1: Detected Flares and ARIMA Model Prediction
        axs[0].plot(time_days, normalized_flux, 'b-', label='Normalized Flux')
        axs[0].plot(time_days, predicted_flux, 'r-', label='ARIMA Model Prediction')
        axs[0].scatter(flare_times, normalized_flux[flare_indices], color='g', label='Detected Flares')
        axs[0].set_title(f'Detected Flares in Lightcurve of TIC {tic_id} using ARIMA')
        axs[0].set_ylabel('Normalized Flux')

# Plot 2: Detected Flares without ARIMA overlay
        axs[1].plot(time_days, normalized_flux, 'b-', label='Normalized Flux')
        axs[1].scatter(flare_times, normalized_flux[flare_indices], color='g', label='Detected Flares')
        axs[1].set_xlabel('Time (days)')
        axs[1].set_ylabel('Normalized Flux')

        handles, labels = axs[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', ncol=4, bbox_to_anchor=(0.5, 0.93), frameon=False)

# Extract TIC ID from the file name
        file_name = os.path.basename(file_path)
        obs_id = file_name.split('-')[2]
        tic_id = obs_id.split('_')[-1]  # Extracting TIC ID from the observation ID
        
# Save the detected flare times to a .txt file
        flare_times_file = os.path.join(r'C:\Users\aadis\Desktop\FLARIMA Flare Times', f'flare_times_TIC{tic_id}.txt')
        np.savetxt(flare_times_file, flare_times)

# Save the plot with TIC ID in the file name
        plot_file_path = os.path.join(r'C:\Users\aadis\Desktop\FLARIMA Plots', f'flare_detection_TIC{tic_id}.png')
# Save the plot as an image
        plt.savefig(plot_file_path)
        plt.show()

    return normalized_flux, time_days, file_path

# Example usage
directory = r'C:\Users\aadis\Desktop\Sector 1'  # Directory containing TESS data files
normalized_flux, time_days, file_path = analyze_tess_data_from_directory(directory)

