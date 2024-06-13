# main.py
import logging
import multiprocessing
import os
from data_processing import analyze_tess_data_from_files

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define file paths (update these as needed)
directory = r'C:\Users\aadis\Downloads\comparison'
trf_file = r'C:\Users\aadis\Downloads\tess-response-function-v2.0.csv'
pecaut_mamajek_file = r'C:\Users\aadis\Downloads\PecautMamajek2013.txt'

def process_subset(files_subset, trf_file, pecaut_mamajek_file, lock):
    """Process a subset of lightcurve files."""
    analyze_tess_data_from_files(files_subset, trf_file, pecaut_mamajek_file, lock)

if __name__ == "__main__":
    # List all lightcurve files in the directory
    all_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('_lc.fits')]

    # Number of processes (modify if needed)
    num_processes = 12

    # Split the list of files into chunks for each process
    file_chunks = [all_files[i::num_processes] for i in range(num_processes)]

    # Create a Manager for shared Lock
    with multiprocessing.Manager() as manager:
        lock = manager.Lock()

        # Create partial function with the lock
        from functools import partial
        process = partial(process_subset, trf_file=trf_file, pecaut_mamajek_file=pecaut_mamajek_file, lock=lock)

        # Use multiprocessing to process each chunk
        with multiprocessing.Pool(processes=num_processes) as pool:
            pool.map(process, file_chunks)