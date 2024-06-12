# utils.py

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