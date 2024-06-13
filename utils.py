# utils.py
'''
utility functions for reading TESS response function data, Pecaut & Mamajek table data for stellar radius
estimation, converting time values, and estimating stellar radii based on temperature using interpolation.

Functions:
----------
1. read_tess_response_function(file_path)
    Reads the TESS response function from a CSV file.

2. read_pecaut_mamajek_table(file_path)
    Reads the Pecaut & Mamajek table for stellar radius estimation based on effective temperature (T_eff).

3. convert_to_bjd(btjd_values, BTJD_OFFSET)
    Converts BTJD (Barycentric TESS Julian Date) to BJD (Barycentric Julian Date) by applying a given offset.

4. estimate_radius_from_teff(teff, pecaut_mamajek_data)
    Estimates the stellar radius from T_eff using the Pecaut & Mamajek table data. If T_eff is not directly available, interpolation is used.
'''

import numpy as np
import pandas as pd
import logging
from scipy.interpolate import interp1d

def read_tess_response_function(file_path):
    try:
        df = pd.read_csv(file_path, sep=',', comment='#', names=['Wavelength (nm)', 'Transmission'], low_memory=False)
        wavelengths = df['Wavelength (nm)'].values
        response = df['Transmission'].values
        return wavelengths, response
    except Exception as e:
        logging.error(f"Error reading TESS response function: {e}")
        raise

def read_pecaut_mamajek_table(file_path):
    pecaut_mamajek_data = {}
    with open(file_path, 'r') as f:
        for line in f:
            if line.strip() and not line.startswith('#'):
                parts = line.split()
                if len(parts) > 14 and parts[0] != 'SpT' and '...' not in parts:  # Avoid the header and invalid data
                    try:
                        temp = float(parts[1])
                        radius = float(parts[13])
                        pecaut_mamajek_data[temp] = radius
                    except ValueError:
                        continue
    return pecaut_mamajek_data

def convert_to_bjd(btjd_values, BTJD_OFFSET):
    return btjd_values + BTJD_OFFSET

def estimate_radius_from_teff(teff, pecaut_mamajek_data):
    temps = np.array(list(pecaut_mamajek_data.keys()))
    radii = np.array(list(pecaut_mamajek_data.values()))
    if teff in temps:
        return pecaut_mamajek_data[teff]
    else:
        # Interpolate to find the radius for the given temperature
        radius_interp = interp1d(temps, radii, kind='linear', fill_value='extrapolate')
        return radius_interp(teff)
