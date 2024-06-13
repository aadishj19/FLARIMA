# lightcurve_analysis.py

'''
This module includes functions for analyzing light curves, specifically aimed at detecting and characterizing stellar flares.
It includes utilities for Planck function computation, integrating luminosity, fitting exponential decay models, and calculating
equivalent durations.

Functions:
----------
1. planck_function(wavelength, temperature)
    Computes the Planck function for a given wavelength and temperature.

2. integrate_luminosity(R_lambda, B_lambda, wavelengths)
    Integrates the luminosity using the TESS response function R_lambda and the Planck function B_lambda over the given wavelengths.

3. exponential_decay(t, A, t0, tau, C)
    Defines an exponential decay model function.

4. fit_exponential_decay(time, flux, initial_guess=(1, 0, 1, 1))
    Fits an exponential decay model to the given time and flux data using non-linear least squares.

5. integrate_exponential_decay(params, start_time, end_time)
    Integrates the exponential decay model over the specified time range.

6. find_flare_times(flare_indices, time, normalized_flux)
    Identifies flare start, peak, and end times based on the flare indices, time array, and normalized flux.

7. equivalent_duration(time, normalized_flux, start, stop, err=False)
    Calculates the equivalent duration of a flare, optionally with error estimation.
'''


import numpy as np
from scipy.optimize import curve_fit
from scipy.integrate import simps, quad
from astropy.constants import sigma_sb
from constants import h, c, k, days_to_seconds, R_sun_to_cm, BTJD_OFFSET
import logging

# Planck Function
def planck_function(wavelength, temperature):
    wavelength_cm = wavelength * 1e-7  # Convert nm to cm
    exp_arg = (h * c) / (wavelength_cm * k * temperature)
    with np.errstate(over='ignore'):  # Ignore overflow warnings
        exp_factor = np.exp(np.clip(exp_arg, None, 700))  # Clip the exponent to a reasonable maximum
    return (2 * h * c ** 2 / (wavelength_cm ** 5)) / (exp_factor - 1)

# Integrate Luminosity using TESS Response Function
def integrate_luminosity(R_lambda, B_lambda, wavelengths):
    return simps(R_lambda * B_lambda, wavelengths)

# Exponential Decay Function
def exponential_decay(t, A, t0, tau, C):
    exp_argument = np.clip(-(t - t0) / tau, a_min=None, a_max=20)  # Prevent overflow
    return A * np.exp(exp_argument) + C

# Fit Exponential Decay
def fit_exponential_decay(time, flux, initial_guess=(1, 0, 1, 1)):
    bounds = ([0, time.min() - 1, 0, 0], [np.inf, time.max() + 1, np.inf, np.inf])
    params, _ = curve_fit(exponential_decay, time, flux, p0=initial_guess, bounds=bounds)
    return params

# Integrate Exponential Decay
def integrate_exponential_decay(params, start_time, end_time):
    A, t0, tau, C = params
    integral, _ = quad(lambda t: exponential_decay(t, A, t0, tau, C), start_time, end_time)
    return integral

# Find flare times
def find_flare_times(flare_indices, time, normalized_flux):
    flare_start_times = []
    flare_end_times = []
    flare_peak_times = []

    for index in flare_indices:
        start_index = index
        while start_index > 0 and normalized_flux[start_index] > normalized_flux[start_index - 1]:
            start_index -= 1
        flare_start_times.append(time[start_index])

        peak_flux = normalized_flux[index]
        flare_peak_times.append(time[index])

        pre_flare_background_flux = normalized_flux[start_index]
        half_max_level = (peak_flux + pre_flare_background_flux) / 2
        end_index = index
        while end_index < len(normalized_flux) - 1 and normalized_flux[end_index] > half_max_level:
            end_index += 1
        flare_end_times.append(time[end_index])

    return flare_start_times, flare_end_times, flare_peak_times

def equivalent_duration(time, normalized_flux, start, stop, err=False):
    try:
        start, stop = int(start), int(stop) + 1
        flare_time_segment = time[start:stop]
        flare_flux_segment = normalized_flux[start:stop]

        residual = flare_flux_segment - 1.0  # Assuming flux is already normalized
        logging.debug(f'Residual: {residual}')
        
        # Ensure there are no NaNs or infinities
        if np.any(np.isnan(residual)) or np.any(np.isinf(residual)):
            raise ValueError("Residual contains NaNs or infinite values")
        
        # Ensure residuals are positive or zero
        residual = np.maximum(residual, 0)
        
        # Convert time to seconds
        x_time_seconds = flare_time_segment * 60.0 * 60.0 * 24.0  
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
