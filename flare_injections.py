#!/usr/bin/env python
# coding: utf-8

# In[ ]:
import numpy as np
import matplotlib.pyplot as plt
from lightkurve import TessLightCurveFile
from astropy.io import fits
from data_processing import analyze_tess_data_from_files  # Import your detection function
import random

def flare_model_davenport2014(t, tpeak, fwhm, ampl, upsample=False, uptime=10):
    _fr = [1.00000, 1.94053, -0.175084, -2.24588, -1.12498]
    _fd = [0.689008, -1.60053, 0.302963, -0.278318]

    fwhm = float(fwhm)
    tpeak = float(tpeak)

    if upsample:
        dt = np.nanmedian(np.diff(t))
        timeup = np.linspace(min(t) - dt, max(t) + dt, t.size * uptime)

        flareup = np.piecewise(
            timeup,
            [
                (timeup <= tpeak) & ((timeup - tpeak) / fwhm > -1.0),
                (timeup > tpeak),
            ],
            [
                lambda x: (
                    _fr[0]
                    + _fr[1] * ((x - tpeak) / fwhm)
                    + _fr[2] * ((x - tpeak) / fwhm) ** 2.0
                    + _fr[3] * ((x - tpeak) / fwhm) ** 3.0
                    + _fr[4] * ((x - tpeak) / fwhm) ** 4.0
                ),
                lambda x: (
                    _fd[0] * np.exp(((x - tpeak) / fwhm) * _fd[1])
                    + _fd[2] * np.exp(((x - tpeak) / fwhm) * _fd[3])
                ),
            ],
        ) * np.abs(ampl)

        flare = np.interp(t, timeup, flareup)
    else:
        flare = np.piecewise(
            t,
            [(t <= tpeak) & ((t - tpeak) / fwhm > -1.0), (t > tpeak)],
            [
                lambda x: (
                    _fr[0]
                    + _fr[1] * ((x - tpeak) / fwhm)
                    + _fr[2] * ((x - tpeak) / fwhm) ** 2.0
                    + _fr[3] * ((x - tpeak) / fwhm) ** 3.0
                    + _fr[4] * ((x - tpeak) / fwhm) ** 4.0
                ),
                lambda x: (
                    _fd[0] * np.exp(((x - tpeak) / fwhm) * _fd[1])
                    + _fd[2] * np.exp(((x - tpeak) / fwhm) * _fd[3])
                ),
            ],
        ) * np.abs(ampl)

    return flare

def flare_model(model, *params):
    if model == "davenport2014":
        return flare_model_davenport2014(*params)
    else:
        raise ValueError(f"Unknown flare model: {model}")

def select_random_flare_times(time, num_flares, min_separation):
    selected_times = []
    while len(selected_times) < num_flares:
        t0 = random.uniform(time.min(), time.max())
        if all(abs(t0 - existing) >= min_separation for existing in selected_times):
            selected_times.append(t0)
    return selected_times

def is_flare_detected(t0, detected_flares, threshold=10 / 60 / 24):  # 10 minutes in days
    for detected in detected_flares:
        if np.abs(detected - t0) < threshold:
            return True
    return False

def injection_grid(lcf, num_flares, flare_fwhms, flare_amplitudes, trf_file, pecaut_mamajek_file):
    flux = lcf.flux.value
    time = lcf.time.value
    quality = np.zeros_like(flux, dtype=int)

    median_flux = np.median(flux)
    std_flux = np.std(flux)

    mask = np.isfinite(flux) & np.isfinite(time)
    flux = flux[mask]
    time = time[mask]
    quality = quality[mask]
    min_separation = max(flare_fwhms) * 2  # Ensure flares do not overlap
    recovered = np.zeros((len(flare_amplitudes), len(flare_fwhms)))

    for i, amplitude in enumerate(flare_amplitudes):
        for j, fwhm in enumerate(flare_fwhms):
            injected_flux = flux.copy()
            flare_times = select_random_flare_times(time, num_flares, min_separation)
            for t0 in flare_times:
                print(f"Injecting flare at tpeak={t0}, FWHM={fwhm}, amplitude={amplitude}")
                normalized_amplitude = amplitude * median_flux / std_flux
                injected_flux += flare_model("davenport2014", time, t0, fwhm, normalized_amplitude)

            # Visualize and save the plot
            plt.figure(figsize=(12, 6))
            plt.plot(time, flux, label='Original Light Curve')
            plt.plot(time, injected_flux, label='Injected Light Curve', alpha=0.6)
            plt.legend()
            plt.xlabel('Time')
            plt.ylabel('Flux')
            plt.title(f'Original vs Injected Light Curve (A={amplitude}, FWHM={fwhm})')
            plot_filename = f'injected_lc_A{amplitude}_FWHM{fwhm}.png'
            plt.savefig(plot_filename)
            plt.close()

            # Save the injected light curve to a temp file
            temp_filename = 'temp_lc_with_flares.fits'
            primary_hdu = fits.PrimaryHDU()
            col1 = fits.Column(name='time', format='D', array=time)
            col2 = fits.Column(name='flux', format='D', array=injected_flux)
            col3 = fits.Column(name='quality', format='I', array=quality)  # Add quality column
            cols = fits.ColDefs([col1, col2, col3])
            hdu = fits.BinTableHDU.from_columns(cols)
            hdul = fits.HDUList([primary_hdu, hdu])
            hdul.writeto(temp_filename, overwrite=True)

            # Analyze the temp file
            detected_flares = analyze_tess_data_from_files([temp_filename], trf_file, pecaut_mamajek_file)  # Wrap the temp_filename in a list
            print(f"Detected flares: {detected_flares}")

            # Check if injected flares are detected
            for t0 in flare_times:
                if is_flare_detected(t0, detected_flares):
                    recovered[i, j] += 1

            recovered[i, j] /= num_flares

    return recovered

# Define your file paths for the necessary files
trf_file = r'C:\Users\aadis\Downloads\tess-response-function-v2.0.csv'
pecaut_mamajek_file = r'C:\Users\aadis\Downloads\PecautMamajek2013.txt'

file_path = r'C:\Users\aadis\Downloads\FLARIMA-main\Test\tess2018234235059-s0002-0000000155776875-0121-s_lc.fits'
lcf = TessLightCurveFile(file_path)

# Define grid with finer scales for better resolution
num_flares = 3  # Number of flares to inject randomly
flare_fwhms = np.linspace(0.002, 0.09, 5)  # in days, finer resolution (2.88 mins to 129.6 mins)
flare_amplitudes = np.linspace(0.1, 2.0, 5)  # in normalized flux units, higher amplitude for testing

# Run injection
recovery_map = injection_grid(lcf, num_flares, flare_fwhms, flare_amplitudes, trf_file, pecaut_mamajek_file)

# Create heatmap plot
plt.figure(figsize=(10, 8))
# Use pcolormesh for better control over binsize
X, Y = np.meshgrid(flare_fwhms, flare_amplitudes)
plt.pcolormesh(X, Y, recovery_map, cmap='heat', shading='auto')  # Change colormap to 'plasma'

plt.colorbar(label='Recovery Probability')
plt.xlabel('Injected FWHM [days]')
plt.ylabel('Injected Amplitude (normalized units)')
plt.title('Flare Recovery Probability')

plt.show()
