#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
from tqdm import tqdm
from astropy.timeseries import LombScargle
from astropy.stats import sigma_clip
from lightkurve import TessLightCurveFile, TessTargetPixelFile
from statsmodels.tsa.arima.model import ARIMA
from astroquery.mast import Observations
from astroquery.exceptions import InvalidQueryError

def download_and_analyze_tess_data(observation_id, download_dir):
    # Query observations and download products
    obs_table = Observations.query_criteria(obs_id=observation_id)
    if len(obs_table) < 1:
        raise InvalidQueryError("Observation list is empty, no associated products.")
    
    data_products = Observations.get_product_list(obs_table[0]['obsid'])
    manifest = Observations.download_products(data_products, download_dir=download_dir)
    
    lc_file_paths = [m['Local Path'] for m in manifest if m['Local Path'].endswith('_lc.fits')] 
    
    for file_path in lc_file_paths:
        lcf = TessLightCurveFile(file_path)  
        pdcsap_flux = lcf.PDCSAP_FLUX.flux
        time = lcf.PDCSAP_FLUX.time.value
        time_days = (time - time[0])
        normalized_flux = (pdcsap_flux - np.mean(pdcsap_flux)) / np.std(pdcsap_flux)

    # Detrend the data using cubic splines
    knots = np.linspace(time_days.min(), time_days.max(), 20)
    spl = UnivariateSpline(time_days, normalized_flux, k=3, s=0)
    detrended_flux = normalized_flux - spl(time_days)
    
    # Hyperparameter Optimization for ARIMA model
    p_values = range(0, 5)
    d_values = range(0, 3)
    q_values = range(0, 5)

    total_iterations = len(p_values) * len(d_values) * len(q_values)
    progress_bar = tqdm(total=total_iterations, desc="Searching for best ARIMA model", position=0)

    best_aic = float("inf")
    best_model = None

    for p in p_values:
        for d in d_values:
            for q in q_values:
                progress_bar.update(1)  # Update the progress bar
                try:
                    model = ARIMA(normalized_flux, order=(p, d, q))
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
    flare_threshold = np.std(residuals) * 4
    flare_indices = np.where(residuals > flare_threshold)[0]
    flare_times = time_days[flare_indices]

    # Print flare times
    print("Detected Flare Times:")
    print(flare_times)

    # Plotting
    fig, axs = plt.subplots(3, 1, figsize=(12, 18), sharex=True, gridspec_kw={'hspace': 0})

    # Plot Detected Flares and ARIMA Model Prediction
    axs[0].plot(time_days, normalized_flux, 'b-', label='Normalized Flux')
    axs[0].plot(time_days, predicted_flux, 'r-', label='ARIMA Model Prediction')
    axs[0].scatter(flare_times, normalized_flux[flare_indices], color='g', label='Detected Flares')
    axs[0].set_title(f'Detected Flares in Lightcurve of TIC {tic_id} using ARIMA')
    axs[0].set_ylabel('Normalized Flux')

    # Plot Detected Flares with Threshold Line
    axs[1].plot(time_days, normalized_flux, 'b-', label='Normalized Flux')
    axs[1].plot(time_days, predicted_flux, 'r-', label='ARIMA Model Prediction')
    axs[1].scatter(flare_times, normalized_flux[flare_indices], color='g', label='Detected Flares')
    axs[1].axhline(flare_threshold, color='r', linestyle='--', label='Threshold')
    axs[1].set_ylabel('Normalized Flux')

    # Plot Detected Flares without ARIMA overlay
    axs[2].plot(time_days, normalized_flux, 'b-', label='Normalized Flux')
    axs[2].scatter(flare_times, normalized_flux[flare_indices], color='g', label='Detected Flares')
    axs[2].set_xlabel('Time (days)')
    axs[2].set_ylabel('Normalized Flux')

    handles, labels = axs[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=4, bbox_to_anchor=(0.5, 0.93), frameon=False)
    plt.savefig(r'C:\Users\aadis\Desktop\FLARIMA Plots\flare_detection_{tic_id}.png')  # Save the plot as an image
    plt.show()

    return normalized_flux, time_days, file_path

observation_id = "tess2019006130736-s0007-0000000266744225-0131-s"
tic_id = observation_id.split('0000000')[1].split('-')[0]
download_dir = r'C:\Users\aadis\Downloads\MAST'
normalized_flux, time_days, file_path = download_and_analyze_tess_data(observation_id, download_dir)

