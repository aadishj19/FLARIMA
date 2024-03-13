#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
from scipy.interpolate import UnivariateSpline
from lightkurve import TessLightCurveFile
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from tqdm import tqdm
from astroquery.mast import Observations, Tesscut
import astropy.units as u
import os
from astropy.table import Table
from astropy.io import fits
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning

def analyze_tess_data_from_directory(directory, resume_last_processed=True):
    lc_file_paths = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('_lc.fits')]

    last_processed_file_path = None
    if resume_last_processed:
        try:
            with open("last_processed_file.txt", "r") as file:
                last_processed_file_path = file.read()
        except FileNotFoundError:
            pass

    if last_processed_file_path in lc_file_paths:
        resume_index = lc_file_paths.index(last_processed_file_path)
        lc_file_paths = lc_file_paths[resume_index + 1:]

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
                    progress_bar.update(1)  
                    try:
                        model = ARIMA(normalized_flux, order=(p, d, q))
                        with warnings.catch_warnings():
                            warnings.filterwarnings("ignore")  
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

        # Initialize additional variables for flare detection parameters
        start_indices = []
        end_indices = []
        peak_indices = []

        # Iterate through flare indices to find start, peak, and end points
        for i in range(len(flare_indices)):
            if i == 0:
                # First flare index, consider it as the start point
                start_index = flare_indices[i]
            elif flare_indices[i] - flare_indices[i-1] > 1:
                # New flare index with a gap, consider it as the start point
                start_index = flare_indices[i]
            else:
                # Consecutive flare index, continue considering it as the start point
                continue

            # Find the peak index by searching for the maximum flux value within the flare
            peak_index = np.argmax(normalized_flux[start_index:flare_indices[i]+1]) + start_index

            # Find the end index by searching for the first index after the peak that has a flux value lower than the start flux value
            for j in range(peak_index+1, len(normalized_flux)):
                if normalized_flux[j] < normalized_flux[start_index]:
                    end_index = j
                    break

            # Check if all required parameters are found
            if start_index is not None and end_index is not None and peak_index is not None:
                # Add the indices to the respective lists
                start_indices.append(start_index)
                end_indices.append(end_index)
                peak_indices.append(peak_index)

        # Apply additional flare detection criteria
        min_duration = 0.05  # Minimum flare duration in days
        min_amplitude = 0.0007  # Minimum flare amplitude

        # Filter out flares that do not meet the criteria
        filtered_indices = []
        for start, peak, end in zip(start_indices, peak_indices, end_indices):
            duration = time_days[end] - time_days[start]
            amplitude = normalized_flux[peak] - normalized_flux[start]
            decline_duration = time_days[end] - time_days[peak]
            rise_duration = time_days[peak] - time_days[start]

            if duration >= min_duration and amplitude >= min_amplitude and decline_duration > rise_duration:
                filtered_indices.append((start, peak, end))

        # Extract the times for the filtered flares
        filtered_flare_times = [time_days[start:end+1] for start, peak, end in filtered_indices]

        # Plotting
        fig, axs = plt.subplots(2, 1, figsize=(12, 18), sharex=True, gridspec_kw={'hspace': 0})

        # Extract TIC ID from the file name
        file_name = os.path.basename(file_path)
        obs_id = file_name.split('-')[2]
        tic_id = obs_id.split('0000000')[1].split('-')[0]  # Extracting TIC ID from the observation ID

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

        # Save the detected flare times to a .txt file
        flare_times_file = os.path.join(r'C:\Users\aadis\Desktop\FLARIMA Flare Times', f'flare_times_TIC{tic_id}.txt')
        np.savetxt(flare_times_file, filtered_flare_times)

        # Save the plot with TIC ID in the file name
        plot_file_path = os.path.join(r'C:\Users\aadis\Desktop\FLARIMA Plots', f'flare_detection_TIC{tic_id}.png')
        # Save the plot as an image
        plt.savefig(plot_file_path)
        plt.show()

        # Update the last processed file
        with open("last_processed_file.txt", "w") as file:
            file.write(file_path)

    return normalized_flux, time_days, file_path

# Example usage
directory = r'C:\Users\aadis\Desktop\Sector 1'  # Directory containing TESS data files
normalized_flux, time_days, file_path = analyze_tess_data_from_directory(directory)
