#!/usr/bin/env python
# coding: utf-8

# In[ ]:
# Libraries and necessary imports
import numpy as np
import os
import matplotlib.pyplot as plt
from astropy.io import fits
from lightkurve import TessLightCurveFile
from statsmodels.tsa.arima.model import ARIMA
import warnings
from tqdm import tqdm
import logging
import csv

# Function definitions
def find_flare_times(flare_indices, time_days, normalized_flux):
    flare_start_times = []
    flare_end_times = []
    flare_peak_times = []

    for index in flare_indices:
        start_index = index
        while start_index > 0 and normalized_flux[start_index] > normalized_flux[start_index - 1]:
            start_index -= 1
        flare_start_times.append(time_days[start_index])

        peak_flux = normalized_flux[index]
        flare_peak_times.append(time_days[index])

        pre_flare_background_flux = normalized_flux[start_index]
        half_max_level = (peak_flux + pre_flare_background_flux) / 2
        end_index = index
        while end_index < len(normalized_flux) - 1 and normalized_flux[end_index] > half_max_level:
            end_index += 1
        flare_end_times.append(time_days[end_index])

    return flare_start_times, flare_end_times, flare_peak_times

# Initialize logging
logging.basicConfig(filename='flare_detection_errors.log', level=logging.ERROR)

def clean_data(time, flux):
    cleaned_flux = np.nan_to_num(flux, nan=0.0, posinf=0.0, neginf=0.0)
    mask = np.isfinite(time) & np.isfinite(cleaned_flux)
    cleaned_time = time[mask]
    cleaned_flux = cleaned_flux[mask]
    return cleaned_time, cleaned_flux

def equivalent_duration(time, flux, start, stop, err=False):
    try:
        start, stop = int(start), int(stop) + 1
        flare_time_segment = time[start:stop] # time in days
        flare_flux_segment = flux[start:stop]

        residual = flare_flux_segment - 1.0  # As flux is normalized
        logging.debug(f'Residual: {residual}')
        
        # Ensure non-negative residuals
        residual = np.maximum(residual, 0)
        
        # Convert time to seconds 
        x_time_seconds = flare_time_segment * 60.0 * 60.0 * 24.0  # days to seconds
        ed = np.sum(np.diff(x_time_seconds) * residual[:-1])
        logging.debug(f'Calculated equivalent duration: {ed}')

        if err:
            flare_chisq = chi_square(residual[:-1], flare_flux_segment.std())
            ederr = np.sqrt(ed**2 / (stop-1-start) / flare_chisq)
            return ed, ederr
        else:
            return ed
    except Exception as e:
        logging.error(f"Error in equivalent_duration: {e}")
        print(f"Error in equivalent_duration: {e}")
        return np.nan

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
        flux = lcf.flux.value  # Convert to correct unit
        time = lcf.time.value  # Convert to numpy array, time in days
        time_days = time
        median_flux = np.median(flux)
        std_flux = np.std(flux)
        normalized_flux = (flux - median_flux) / std_flux
        normalized_flux = normalized_flux + 1

        # Reevaluate ARIMA Model Parameters
        p_values = range(0, 3)
        d_values = range(0, 2)
        q_values = range(0, 3)

        # Hyperparameter Optimization
        total_iterations = len(p_values) * len(d_values) * len(q_values)
        progress_bar = tqdm(total=total_iterations, desc="Searching for best ARIMA model", position=0)

        best_aic = float("inf")
        best_model = None

        warnings.filterwarnings("ignore", category=Warning)  # Ignore general warnings

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
        flare_threshold = np.std(residuals) * 3.5
        flare_indices = np.where(residuals > flare_threshold)[0]
        flare_times = time_days[flare_indices]

        # Find flare times
        flare_start_times, flare_end_times, flare_peak_times = find_flare_times(flare_indices, time_days, normalized_flux)

        # Extract TIC ID from the file name
        file_name = os.path.basename(file_path)
        obs_id = file_name.split('-')[2]
        tic_id = obs_id.split('0000000')[1].split('-')[0]  # Extracting TIC ID from the observation ID

        # Apply additional flare detection criteria
        min_duration = 0.004  # Minimum flare duration in days
        min_amplitude = 0.0005  # Minimum flare amplitude

        # Filter out flares that do not meet the criteria
        filtered_flare_times = []
        filtered_amplitudes = []
        for start_time, peak_time, end_time in zip(flare_start_times, flare_peak_times, flare_end_times):
            start_index = np.argmin(np.abs(time_days - start_time))
            peak_index = np.argmin(np.abs(time_days - peak_time))

            amplitude = normalized_flux[peak_index] - normalized_flux[start_index]
            duration = end_time - start_time  # duration in days
            decline_duration = end_time - peak_time  # in days
            rise_duration = peak_time - start_time  # in days
            points_between = len(np.where((time_days >= start_time) & (time_days <= end_time))[0])

            if duration >= min_duration and amplitude >= min_amplitude and decline_duration > rise_duration and points_between >= 2:
                filtered_flare_times.append((start_time, end_time, peak_time))
                filtered_amplitudes.append(amplitude)
            else:
                print("Criteria not met for this potential flare.")

        # Compute equivalent durations for each detected flare
        flare_equivalent_durations = []
        for (start_time, end_time, peak_time), amplitude in zip(filtered_flare_times, filtered_amplitudes):
            start_index = np.argmin(np.abs(time_days - start_time))
            end_index = np.argmin(np.abs(time_days - end_time))

            print(f"Calculating equivalent duration for flare from {start_time} to {end_time}")

            try:
                ed = equivalent_duration(time_days, normalized_flux, start_index, end_index)
                flare_equivalent_durations.append(ed)
                print(f"Calculated equivalent duration: {ed}")
            except Exception as e:
                logging.error(f"Error calculating equivalent duration for flare from {start_time} to {end_time}: {e}")
                flare_equivalent_durations.append(np.nan)

        # Plotting
        original_flux = flux
        fig, axs = plt.subplots(3, 1, figsize=(12, 24), sharex=True, gridspec_kw={'hspace': 0.3})

        # Plot 1: Original Flux with Filtered Detected Flares
        axs[0].plot(time_days, original_flux, 'b-', label='Original Flux')
        filtered_peak_times = [peak_time for start_time, end_time, peak_time in filtered_flare_times]
        filtered_peak_fluxes = [original_flux[np.argmin(np.abs(time_days - peak_time))] for peak_time in filtered_peak_times]
        axs[0].scatter(filtered_peak_times, filtered_peak_fluxes, color='r', label='Filtered Flares', zorder=3)
        axs[0].set_title(f'Filtered Flares in Original Lightcurve of TIC {tic_id}')
        axs[0].set_ylabel('Flux')
        axs[0].legend()

        # Plot 2: Normalized Flux with ARIMA Model Prediction
        axs[1].plot(time_days, normalized_flux, 'b-', label='Normalized Flux')
        axs[1].plot(time_days, predicted_flux, 'r-', label='ARIMA Model Prediction')
        axs[1].scatter(flare_times, normalized_flux[flare_indices], color='g', label='Detected Flares')
        axs[1].set_title(f'Detected Flares in Lightcurve of TIC {tic_id} using ARIMA')
        axs[1].set_ylabel('Normalized Flux')
        axs[1].legend()

        # Plot 3: Detected Flares that meet the additional criteria
        filtered_flux_values = [float(normalized_flux[np.where(time_days == peak_time)][0]) for peak_time in filtered_peak_times]  # Ensure scalar values
        axs[2].plot(time_days, normalized_flux, 'b-', label='Normalized Flux')
        axs[2].scatter(filtered_peak_times, filtered_flux_values, color='r', label='Detected Flares', zorder=3)
        axs[2].set_xlabel('Time (BJD)')
        axs[2].set_ylabel('Normalized Flux')
        axs[2].legend()

        # Save the plot with TIC ID in the file name
        plot_file_path = os.path.join(r'C:\Users\aadis\Desktop\May26_plots', f'flare_detection_TIC{tic_id}.png')
        plt.savefig(plot_file_path)

        # Extract flare properties and save to CSV
        headers = ['tess_id', 'flare_start_time', 'flare_end_time', 'flare_peak_time', 'amplitude', 'duration', 'equivalent_duration']
        csv_file_path = os.path.join(r'C:\Users\aadis\Desktop\May26_plots', "flare_results_may_26.csv")
        file_exists = os.path.isfile(csv_file_path)

        with open(csv_file_path, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            # Only write headers if the file is newly created
            if not file_exists:
                writer.writerow(headers)

            for ((start_time, end_time, peak_time), amplitude, ed) in zip(filtered_flare_times, filtered_amplitudes, flare_equivalent_durations):
                duration = end_time - start_time  # duration in days
                writer.writerow([tic_id, start_time, end_time, peak_time, amplitude, duration, ed])

        # Update the last processed file
        with open("last_processed_file.txt", "w") as file:
            file.write(file_path)

    return normalized_flux, file_path

# Example usage
directory = r'C:\Users\aadis\Downloads\comparison'  # Directory containing TESS data files

# Run the analysis and save the results to the CSV file
analyze_tess_data_from_directory(directory)
