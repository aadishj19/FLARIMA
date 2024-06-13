# FLARIMA

Code for Characterizing Stellar Flares in TESS Data. The project aims to detect and analyze stellar flares using data from the TESS (Transiting Exoplanet Survey Satellite) mission. The code is designed to process light curve data, detect flares using ARIMA (AutoRegressive Integrated Moving Average) models, and characterize the detected flares.

## Project Structure

The project is organized into several Python modules, each with a specific purpose:
- `constants.py`: Contains all the constants used throughout the project.
- `utils.py`: Contains utility functions for reading and processing data.
- `lightcurve_analysis.py`: Contains functions for analyzing light curves.
- `data_processing.py`: Contains the main data processing loop.
- `main.py`: The main script to run the analysis.

## Directory Structure
project/
│
├── constants.py
├── utils.py
├── lightcurve_analysis.py
├── data_processing.py
├── main.py
└── last_processed_file.txt (optional, if resuming from the last processed file)


## Installation

To run this project, you need to have Python installed along with some specific libraries. You can install the required packages using :
```
pip install -r requirements.txt
```
**Usage**
1. Setup
Ensure you have the necessary TESS light curve data files in your working directory. You will also need the TESS response function file and the Pecaut & Mamajek table. Update the paths in your main.py as well as data_processing.py as needed.

2. Running the Analysis
Execute the main.py script to start the analysis process:
Processes TESS light curve files, detects flares, and calculates their properties.

**Results**
The results include:
Plots of the detected flares and their properties.
CSV file containing the properties of the detected flares, such as start time, end time, amplitude, duration, and flare energy.


## Note

**Flarima** is still under active development. Future plan is to add flare injection and recovery tests. 
