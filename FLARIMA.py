#!/usr/bin/env python
# coding: utf-8

# In[ ]:
import os
import numpy as np
import warnings
from scipy.interpolate import UnivariateSpline
from statsmodels.tsa.arima.model import ARIMA
from sklearn.exceptions import ConvergenceWarning
import matplotlib.pyplot as plt
from tqdm import tqdm
import statsmodels.api as sm
from astroquery.mast import Observations, Tesscut
import astropy.units as u
from astropy.table import Table
from astropy.io import fits
from scipy.optimize import curve_fit
import csv
import pandas as pd
from lightkurve import TessLightCurveFile
from astropy.utils.data import download_file
import itertools
from astropy.timeseries import LombScargle
import pickle
from scipy.signal import find_peaks
import scipy.optimize as opt
import scipy.integrate as integrate

def find_flare_times(flare_indices, time_days, normalized_flux):
    flare_start_times = []
    flare_end_times = []
    flare_peak_times = []

    for index in flare_indices:
        # Assuming the start time finding mechanism remains unchanged
        start_index = index
        while start_index > 0 and normalized_flux[start_index] > normalized_flux[start_index - 1]:
            start_index -= 1
        flare_start_times.append(time_days[start_index])

        # The peak time is direct; no change needed from original description
        peak_flux = normalized_flux[index]
        flare_peak_times.append(time_days[index])

        # Modified end time finding mechanism
        pre_flare_background_flux = normalized_flux[start_index]  # Approximation of background level from start
        half_max_level = (peak_flux + pre_flare_background_flux) / 2
        end_index = index
        # Move forward until the flux drops below half_max_level
        while end_index < len(normalized_flux) - 1 and normalized_flux[end_index] > half_max_level:
            end_index += 1
        flare_end_times.append(time_days[end_index])

    return flare_start_times, flare_end_times, flare_peak_times

def exponential_decay(t, A, t0, tau, C):
    return A * np.exp(-(t - t0) / tau) + C

def fit_exponential_decay(time, flux, initial_guess=(1, 0, 1, 1)):
    bounds = ([0, time.min(), 0, 0], [np.inf, time.max(), np.inf, np.inf])
    params, cov = opt.curve_fit(exponential_decay, time, flux, p0=initial_guess, bounds=bounds)
    return params

def integrate_exponential_decay(params, start, end):
    A, t0, tau, C = params
    integral, _ = integrate.quad(lambda t: A * np.exp(-(t - t0) / tau) + C, start, end)
    return integral

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

        # Add a constant to detrended flux to set baseline to 1
        baseline = 1 
        normalized_flux += (baseline - np.min(normalized_flux))

        # Reevaluate ARIMA Model Parameters
        p_values = range(0, 3)
        d_values = range(0, 2)
        q_values = range(0, 3)

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
        
        # Find flare times
        flare_start_times, flare_end_times, flare_peak_times = find_flare_times(flare_indices, time_days, normalized_flux)

        # Extract TIC ID from the file name
        file_name = os.path.basename(file_path)
        obs_id = file_name.split('-')[2]
        tic_id = obs_id.split('0000000')[1].split('-')[0]  # Extracting TIC ID from the observation ID
        
        # Apply additional flare detection criteria
        min_duration = 0.002  # Minimum flare duration in days
        min_amplitude = 0.0005  # Minimum flare amplitude

        # Filter out flares that do not meet the criteria
        filtered_flare_times = []
        for start_time, peak_time, end_time in zip(flare_start_times, flare_peak_times, flare_end_times):
            amplitude = normalized_flux[np.where(time_days == peak_time)] - normalized_flux[np.where(time_days == start_time)]
            duration = end_time - start_time
            decline_duration = end_time - peak_time
            rise_duration = peak_time - start_time
            points_between = len(np.where((time_days >= start_time) & (time_days <= end_time))[0])

            if duration >= min_duration and amplitude >= min_amplitude and decline_duration > rise_duration and points_between >= 2:
                filtered_flare_times.append((start_time, end_time, peak_time))
            else:
                print("Criteria not met for this potential flare.")
                
                
        # For each detected flare:
        flare_energies = []
        for start_time, end_time, peak_time in filtered_flare_times:
            flare_time_segment = time_days[(time_days >= start_time) & (time_days <= end_time)]
            flare_flux_segment = normalized_flux[(time_days >= start_time) & (time_days <= end_time)]

            # Fit the exponential decay model to the flare
            initial_guess = (flare_flux_segment.max(), peak_time, 0.1, 1)
            try:
                params = fit_exponential_decay(flare_time_segment, flare_flux_segment, initial_guess)
                energy = integrate_exponential_decay(params, start_time, end_time)
                flare_energies.append(energy)
            except RuntimeError:
                print("Could not fit an exponential decay to this flare.")
                flare_energies.append(np.nan)
                
        # Plotting
        fig, axs = plt.subplots(2, 1, figsize=(12, 18), sharex=True, gridspec_kw={'hspace': 0})
        
        # Plot 1: Detected Flares and ARIMA Model Prediction
        axs[0].plot(time_days, normalized_flux, 'b-', label='Normalized Flux')
        axs[0].plot(time_days, predicted_flux, 'r-', label='ARIMA Model Prediction')
        axs[0].scatter(flare_times, normalized_flux[flare_indices], color='g', label='Detected Flares')
        axs[0].set_title(f'Detected Flares in Lightcurve of TIC {tic_id} using ARIMA')
        axs[0].set_ylabel('Normalized Flux')

        # Plot 2: Detected Flares that meet the additional criteria
        axs[1].plot(time_days, normalized_flux, 'b-', label='Normalized Flux')
        filtered_peak_times = [peak_time for start_time, end_time, peak_time in filtered_flare_times]
        filtered_flux = [normalized_flux[np.where(time_days == peak_time)][0] for peak_time in filtered_peak_times]
        axs[1].scatter(filtered_peak_times, filtered_flux, color='r', label='Detected Flares', zorder=3)  # Scatter plot for detected flares
        axs[1].set_xlabel('Time (BJD)')
        axs[1].set_ylabel('Normalized Flux')

        # Save the plot with TIC ID in the file name
        plot_file_path = os.path.join(r'C:\Users\aadis\Desktop\May Plots', f'flare_detection_TIC{tic_id}.png')
        plt.savefig(plot_file_path)
                
                
        # Extract flare properties and save to CSV
        # Extract flare properties and append them to a CSV file
        headers = ['tess_id', 'flare_start_time', 'flare_end_time', 'flare_peak_time', 'amplitude', 'duration', 'energy']
        csv_file_path = os.path.join(directory, "flare_results_may.csv")
        file_exists = os.path.isfile(csv_file_path)

        with open(csv_file_path, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            # Only write headers if the file is newly created
            if not file_exists:
                writer.writerow(headers)

            for (start_time, end_time, peak_time), energy in zip(filtered_flare_times, flare_energies):
                # Calculate amplitude and duration for the current flare
                start_index = np.argmin(np.abs(time_days - start_time))
                peak_index = np.argmin(np.abs(time_days - peak_time))
                amplitude = normalized_flux[peak_index] - normalized_flux[start_index]
                duration = end_time - start_time

                writer.writerow([tic_id, start_time, end_time, peak_time, amplitude, duration, energy])

# Example usage
directory = r'C:\Users\aadis\Downloads\comparison'  # Directory containing TESS data files

# Create a new CSV file and write the header row
csv_file = os.path.join(directory, "flare_results_may.csv")

# Run the analysis and save the results to the CSV file
analyze_tess_data_from_directory(directory)
