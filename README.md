# Detector Calibration and Characterization Script

This project provides a Python script (`script_done.py`) designed to calibrate and characterize data from three detectors (NaI, CdTe, and BGO). The script processes detector data using specific parameters and regions of interest, as defined by supplementary files in the project directory.

## Project Structure

Ensure that the following structure is maintained, as the script relies on specific file and folder naming conventions:

- **Folders**: Include separate folders for each detector's data. The folder names must align with the script's internal naming scheme.
- **Files**:
  - `Parameter.txt`: Contains essential source information, based on lab data.
  - `ranges_of_interest.txt`: Specifies ranges of interest for data processing. This file is customizable, and you will have an opportunity to adjust values through prompts during execution.

> **Note**: `script_done.py` takes no command-line arguments. Ensure all folders and the `Parameter.txt` and `ranges_of_interest.txt` files are present in the same directory as the script.

## Usage

1. Place all required folders and files in the same directory as `script_done.py`.
2. Run the script without any arguments:
   ```bash
   python script_done.py
