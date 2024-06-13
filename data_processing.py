# data_processing.py
'''
This function processes TESS light curve files, detects potential stellar flares,
and calculates various properties such as their amplitude, duration, and energy. The analysis
includes normalizing the flux, applying an ARIMA model to detect flares, fitting an exponential
decay model for flare energy estimation, and calculating equivalent durations.

    Parameters:
    ----------
    file_paths : list of str
        List of file paths to TESS light curve files to be analyzed.
    trf_file : str
        File path to the TESS response function data.
    pecaut_mamajek_file : str
        File path to the Pecaut & Mamajek table data necessary for radius estimation.
    lock : multiprocessing.Lock
        A lock to ensure safe writing of results in a multi-processing context.

    Processing Steps:
    -----------------
    1. Load the TESS response function and Pecaut & Mamajek table data.
    2. For each TESS light curve file:
       a. Extract flux and time data from the light curve file.
       b. Normalize the flux data.
       c. Read necessary header information (e.g., stellar effective temperature and radius).
       d. If stellar effective temperature (T_eff) is available:
          - Compute the stellar luminosity using Planck's function.
       e. Apply ARIMA models to detect flare indices.
       f. Filter flare candidates based on thresholds.
       g. If stellar radius is available:
          - Fit the exponential decay model to estimate flare energy.
       h. Calculate the equivalent duration of the detected flares.
    3. Save results (flare properties and equivalent durations) to a CSV file.
    4. Generate and save plots illustrating the original flux, ARIMA model predictions, and detected flares.

    Flare Detection:
    ----------------
    - Flares are detected using an ARIMA model residual analysis.
    - Flares are filtered based on duration, amplitude, and other criteria to ensure valid detections.
    - Flare properties include start and end times, peak time (in BJD), amplitude, duration, and energy.

    Plotting:
    ----------
    - Three plots are generated:
      1. Original flux with filtered detected flares.
      2. Normalized flux with ARIMA model prediction and detected flares.
      3. Detected flares that meet additional criteria.

    Output:
    -------
    - CSV file containing detected flare properties.
    - PNG files with plots showing flare detections for each TESS light curve file.
'''
import os
import csv
import logging
import numpy as np
import matplotlib.pyplot as plt
import warnings
from statsmodels.tsa.arima.model import ARIMA
from astropy.constants import sigma_sb
from lightkurve import TessLightCurveFile
from astropy.io import fits
from tqdm import tqdm
from constants import R_sun_to_cm, BTJD_OFFSET, days_to_seconds
from utils import read_tess_response_function, read_pecaut_mamajek_table, convert_to_bjd
from lightcurve_analysis import planck_function, integrate_luminosity, fit_exponential_decay, integrate_exponential_decay, find_flare_times, equivalent_duration

def analyze_tess_data_from_files(file_paths, trf_file, pecaut_mamajek_file, lock):
    wavelengths, R_lambda = read_tess_response_function(trf_file)

    # Read Pecaut & Mamajek Table for radius estimation
    pecaut_mamajek_data = read_pecaut_mamajek_table(pecaut_mamajek_file)

    for file_path in file_paths:
        try:
            lcf = TessLightCurveFile(file_path)
            flux = lcf.flux.value  # Extract correct lightcurve flux
            time = lcf.time.value  # Extract correct time

            if flux.size == 0 or time.size == 0:
                logging.error(f"Empty light curve data in {file_path}")
                continue

            # Normalize flux
            median_flux = np.median(flux)
            std_flux = np.std(flux)
            normalized_flux = (flux - median_flux) / std_flux + 1

            # Extract necessary header information
            header = fits.getheader(file_path)
            teff = header.get('TEFF')
            radius_star = header.get('RADIUS')
                
            # Check if TEFF or RADIUS is missing
            teff_missing = teff is None or not np.isfinite(teff)
            radius_missing = radius_star is None or not np.isfinite(radius_star)
            radius_star_cm = radius_star * R_sun_to_cm if not radius_missing else None

            B_lambda_star = None
            L_star_prime = None

            if not teff_missing:
                B_lambda_star = planck_function(wavelengths, teff)
                L_star_prime = integrate_luminosity(R_lambda, B_lambda_star, wavelengths)

            # ARIMA model to detect flare indices
            p_values = range(0, 3)
            d_values = range(0, 2)
            q_values = range(0, 3)

            best_aic = float("inf")
            best_model = None

            warnings.filterwarnings("ignore", category=Warning)  # Ignore general warnings

            for p in p_values:
                for d in d_values:
                    for q in q_values:
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

            predicted_flux = best_model.predict(start=0, end=len(time) - 1)
            residuals = normalized_flux - predicted_flux

            flare_threshold = np.std(residuals) * 3.5
            flare_indices = np.where(residuals > flare_threshold)[0]
            flare_times = time[flare_indices]
            flare_start_times, flare_end_times, flare_peak_times = find_flare_times(flare_indices, time, normalized_flux)
            
            # Extract TIC ID from the file name
            file_name = os.path.basename(file_path)
            obs_id = file_name.split('-')[2]
            tic_id = obs_id.split('0000000')[1].split('-')[0]  # Extracting TIC ID from the observation ID
            
            min_duration = 0.004  # Minimum flare duration in days
            min_amplitude = 0.0005  # Minimum flare amplitude

            # Calculate properties for each potential flare and filter them
            flare_properties = []
            filtered_flare_times = []
            filtered_peak_times = []

            for start_time, peak_time, end_time in zip(flare_start_times, flare_peak_times, flare_end_times):
                start_index = np.argmin(np.abs(time - start_time))
                peak_index = np.argmin(np.abs(time - peak_time))
                end_index = np.argmin(np.abs(time - end_time))

                pre_flare_flux = np.median(normalized_flux[:start_index])  # Baseline flux before the flare
                peak_flux = normalized_flux[peak_index] # Flux at the peak of the flare
                amplitude = (peak_flux - pre_flare_flux) / pre_flare_flux
                duration = end_time - start_time
                decline_duration = end_time - peak_time
                rise_duration = peak_time - start_time
                points_between = len(np.where((time >= start_time) & (time <= end_time))[0])

                # Process only valid flares based on conditions
                if duration >= min_duration and amplitude >= min_amplitude and decline_duration > rise_duration and points_between >= 2:
                    peak_time_bjd = convert_to_bjd(peak_time, BTJD_OFFSET)
                    peak_time_bjd_formatted = f"{peak_time_bjd:.6f}"

                    filtered_flare_times.append((start_time, end_time, peak_time))
                    filtered_peak_times.append(peak_time)

                    # Calculate the flare energy
                    flare_time_segment = time[(time >= start_time) & (time <= end_time)]
                    flare_flux_segment = normalized_flux[(time >= start_time) & (time <= end_time)]

                    if flare_time_segment.size == 0 or flare_flux_segment.size == 0:
                        logging.warning(f"Empty flare segments in {file_path}: start_time={start_time}, end_time={end_time}")
                        continue

                    initial_guess = (flare_flux_segment.max(), peak_time, 0.1, 1)
                    flare_energy = "N/A"

                    try:
                        params = fit_exponential_decay(flare_time_segment, flare_flux_segment, initial_guess)
                        if not (teff_missing or radius_missing):
                            A_flare = integrate_exponential_decay(params, start_time, end_time)
                            B_lambda_flare = planck_function(wavelengths, 9000)
                            L_flare_prime = integrate_luminosity(R_lambda, B_lambda_flare, wavelengths)
                            A_flare_abs = A_flare * np.pi * radius_star_cm ** 2 * L_star_prime / L_flare_prime
                            flare_luminosity = sigma_sb.value * (9000 ** 4) * A_flare_abs
                            flare_energy = flare_luminosity * (end_time - start_time) * days_to_seconds  # Convert days to seconds
                    except RuntimeError as e:
                        logging.warning(f"Flare fitting error for {file_path} at start_time={start_time}: {e}")
                        continue
                        
                    # Append flare properties only for valid flares
                    flare_properties.append([tic_id, teff, start_time, end_time, peak_time_bjd_formatted, amplitude, duration, flare_energy])

                        
                #Calculating Equivalent Duration        
                flare_equivalent_durations = []
                for start_time, end_time, peak_time in filtered_flare_times:
                    start_index = np.argmin(np.abs(time - start_time))
                    end_index = np.argmin(np.abs(time - end_time))

                    print(f"Calculating equivalent duration for flare from {start_time} to {end_time}")

                    try:
                        ed = equivalent_duration(time, normalized_flux, start_index, end_index)
                        flare_equivalent_durations.append(ed)
                        print(f"Calculated equivalent duration: {ed}")
                    except Exception as e:
                        logging.error(f"Error calculating equivalent duration for flare from {start_time} to {end_time}: {e}")
                        flare_equivalent_durations.append(np.nan)


            # Plotting
            original_flux = flux
            fig, axs = plt.subplots(3, 1, figsize=(12, 24), sharex=True, gridspec_kw={'hspace': 0.3})

            # Plot 1: Original Flux with Filtered Detected Flares
            axs[0].plot(time, original_flux, 'b-', label='Original Flux')
            filtered_peak_times = [peak_time for start_time, end_time, peak_time in filtered_flare_times]
            filtered_peak_fluxes = [original_flux[np.argmin(np.abs(time - peak_time))] for peak_time in filtered_peak_times]
            axs[0].scatter(filtered_peak_times, filtered_peak_fluxes, color='r', label='Filtered Flares', zorder=3)
            axs[0].set_title(f'Filtered Flares in Original Lightcurve of TIC {tic_id}')
            axs[0].set_ylabel('Flux')
            axs[0].legend()

            # Plot 2: Normalized Flux with ARIMA Model Prediction
            axs[1].plot(time, normalized_flux, 'b-', label='Normalized Flux')
            axs[1].plot(time, predicted_flux, 'r-', label='ARIMA Model Prediction')
            axs[1].scatter(flare_times, normalized_flux[flare_indices], color='g', label='Detected Flares')
            axs[1].set_title(f'Detected Flares in Lightcurve of TIC {tic_id} using ARIMA')
            axs[1].set_ylabel('Normalized Flux')
            axs[1].legend()

            # Plot 3: Detected Flares that meet the additional criteria
            axs[2].plot(time, normalized_flux, 'b-', label='Normalized Flux')
            filtered_flux = [normalized_flux[np.where(time == peak_time)][0] for peak_time in filtered_peak_times]
            axs[2].scatter(filtered_peak_times, filtered_flux, color='r', label='Detected Flares', zorder=3)
            axs[2].set_xlabel('Time (BJD)')
            axs[2].set_ylabel('Normalized Flux')
            axs[2].legend()

            # Save the plot with TIC ID in the file name
            plot_file_path = os.path.join(r'C:\Users\aadis\Downloads\FLARIMA-main\FLARIMA-main\plots', f'flare_detection_TIC{tic_id}.png')
            plt.savefig(plot_file_path)
            
            for i, props in enumerate(flare_properties):
                props.append(flare_equivalent_durations[i] if i < len(flare_equivalent_durations) else np.nan)

            # Save results
            headers = ['TIC_ID', 'T_eff', 'Start_time', 'End_time', 'Peak_time(BJD)', 'Amplitude', 'Duration(days)', 'Flare_energy(erg)', 'ED(s)']
            csv_file_path = r'C:\Users\aadis\Downloads\FLARIMA-main\FLARIMA-main\flare_energies.csv'
            file_exists = os.path.isfile(csv_file_path)


            with lock:
                with open(csv_file_path, 'a', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    if not file_exists:
                        writer.writerow(headers)
                    for props in flare_properties:
                        writer.writerow(props)

            # Update last processed file
            #with open("last_processed_file.txt", "w") as file:
                #file.write(file_path)

        except Exception as e:
            logging.error(f"Error processing {file_path}: {e}")
