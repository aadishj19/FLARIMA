# main.py

import logging
from data_processing import analyze_tess_data_from_directory

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define file paths (update these as needed)
directory = r'C:\Users\path_to_directory'
trf_file = r'C:\Users\path_to_directory\tess-response-function-v2.0.csv'
pecaut_mamajek_file = r'C:\Users\path_to_directory\PecautMamajek2013.txt'

if __name__ == "__main__":
    analyze_tess_data_from_directory(directory, trf_file, pecaut_mamajek_file)
