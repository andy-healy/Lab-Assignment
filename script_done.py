## Import libaries 
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timezone
import math
from scipy.optimize import curve_fit
from matplotlib.ticker import MultipleLocator


##Curve fitting functions
# Define the quadratic model for (δE)^2 as a function of E
def quadratic_model(E, a, b, c):
    return a + b * E + c * E**2

def efficiency_fit_function(ln_E, a, b, c):
    """Polynomial function in ln(E) for fitting."""
    return a + b * ln_E + c * (ln_E)**2
    
def fit_linear(x, a, b):
    """Linear fitting function."""
    return a * x + b

def fit_quadratic(x, a, b, c):
    """Quadratic fitting function."""
    return a * x**2 + b * x + c

def fit_exponential(x, a, b):
    """Exponential fitting function."""
    return a * np.exp(b * x)

def gaussian_profile(x, mu, std_dev, A):
    """Function that returns Gaussian Profile of x"""
    return A / (np.sqrt(2 * np.pi) * std_dev) * np.exp(-(x - mu)**2 / (2 * std_dev**2))

def double_gaussian_profile(x, mu1, std_dev1, A1, mu2, std_dev2, A2):
    """Function that returns a double Gaussian profile of x."""
    gaussian1 = A1 / (np.sqrt(2 * np.pi) * std_dev1) * np.exp(-(x - mu1)**2 / (2 * std_dev1**2))
    gaussian2 = A2 / (np.sqrt(2 * np.pi) * std_dev2) * np.exp(-(x - mu2)**2 / (2 * std_dev2**2))
    return gaussian1 + gaussian2

def combined_linear(x, a, b, mu, std_dev, A):
## Fit the Gaussian profile
    return fit_linear(x, a, b) + gaussian_profile(x, mu, std_dev, A)

def combined_quadratic(x, a, b, c, mu, std_dev, A):
## Fit the Gaussian profile
    return fit_quadratic(x, a, b, c) + gaussian_profile(x, mu, std_dev, A)

def combined_exponential(x, a, b, mu, std_dev, A):
## Fit the Gaussian profile
    return fit_exponential(x, a, b) + gaussian_profile(x, mu, std_dev, A)

def combined_linear_double_gaussian(x, a, b, mu1, std_dev1, A1, mu2, std_dev2, A2):
    """Fit the linear function and add double Gaussian profile."""
    return fit_linear(x, a, b) + double_gaussian_profile(x, mu1, std_dev1, A1, mu2, std_dev2, A2)
    
def combined_quadratic_double_gaussian(x, a, b, c, mu1, std_dev1, A1, mu2, std_dev2, A2):
    """Fit the quadratic function and add double Gaussian profile."""
    return fit_quadratic(x, a, b, c) + double_gaussian_profile(x, mu1, std_dev1, A1, mu2, std_dev2, A2)

def combined_exponential_double_gaussian(x, a, b, mu1, std_dev1, A1, mu2, std_dev2, A2):
    """Fit the exponential function and add double Gaussian profile."""
    return fit_exponential(x, a, b) + double_gaussian_profile(x, mu1, std_dev1, A1, mu2, std_dev2, A2)

def fit_combined_quadratic_double(channel, counts, initial_guess_params):
    """Fit combined quadratic and double Gaussian function."""
    mu1, std_dev1, A, mu2, std_dev2, A2 = initial_guess_params

    # Initial guesses: [a, b, c, mu1, std_dev1, A, mu2, std_dev2, A2]
    initial_guess = [1, 1, 1, mu1, std_dev1, A, mu2, std_dev2, A2]
    popt, _ = curve_fit(combined_quadratic_double_gaussian, channel, counts, p0=initial_guess)
    return popt

def fit_combined_exponential_double(channel, counts, initial_guess_params):
    """Fit combined exponential and double Gaussian function."""
    mu1, std_dev1, A, mu2, std_dev2, A2 = initial_guess_params

    # Initial guesses: [a, b, mu1, std_dev1, A, mu2, std_dev2, A2]
    initial_guess = [1, 0.01, mu1, std_dev1, A, mu2, std_dev2, A2]
    popt, _ = curve_fit(combined_exponential_double_gaussian, channel, counts, p0=initial_guess)
    return popt

def fit_combined_quadratic(channel, counts, initial_guess_params):
    """Fit combined quadratic and Gaussian function."""
    mu, std_dev, A = initial_guess_params

    # Initial guesses: [a, b, c, mu, std_dev, A]
    initial_guess = [1, 1, 1, mu, std_dev, A]
    popt, _ = curve_fit(combined_quadratic, channel, counts, p0=initial_guess)
    return popt

def fit_combined_linear(channel, counts, initial_guess_params):
    """Fit combined linear and Gaussian function."""
    mu, std_dev, A = initial_guess_params

    # Initial guesses: [a, b, mu, std_dev, A]
    initial_guess = [1, 1, mu, std_dev, A]
    popt, _ = curve_fit(combined_linear, channel, counts, p0=initial_guess)
    return popt

def fit_combined_exponential(channel, counts, initial_guess_params):
    """Fit combined exponential and Gaussian function."""
    mu, std_dev, A = initial_guess_params

    # Initial guesses: [a, b, mu, std_dev, A]
    initial_guess = [1, 0.01, mu, std_dev, A]
    popt, _ = curve_fit(combined_exponential, channel, counts, p0=initial_guess)
    return popt

def fit_combined_linear_double(channel, counts, initial_guess_params):
    """Fit combined linear and double Gaussian function."""
    mu1, std_dev1, A, mu2, std_dev2, A2 = initial_guess_params

    # Initial guesses: [a, b, mu1, std_dev1, A, mu2, std_dev2, A2]
    initial_guess = [1, 1, mu1, std_dev1, A, mu2, std_dev2, A2]
    popt, _ = curve_fit(combined_linear_double_gaussian, channel, counts, p0=initial_guess)
    return popt

## Reads and stores all data files


## Parses the spe files
def read_measurement_data_spe(filename, unit):
    with open(filename, 'r') as file:
        lines = file.readlines()

    date_measured = "" 
    meas_tim = 0.0
    data_values = []
    data_section_started = False
    first_data_line = False

    for i, line in enumerate(lines):
        line = line.strip()

        ## Extract date and measurement time
        if line.startswith('$DATE_MEA:'):
            date_measured = lines[i + 1].strip()
            date_measured = parse_datetime_with_datetime_module(date_measured)
        elif line.startswith('$MEAS_TIM:'):
            # Handle the case where the line might have multiple values
            try:
                meas_tim_value = lines[i + 1].strip().split()[0]  # Get the first value
                meas_tim = float(meas_tim_value)
            except ValueError:
                print(f"Warning: Could not convert measurement time to float: '{lines[i + 1].strip()}'")
        elif line.startswith('$DATA:'):
            data_section_started = True
            first_data_line = True
            continue

        # Data section
        if data_section_started:
            if line.startswith('$'):  # If a new section starts, stop reading data
                break
            if first_data_line:
                first_data_line = False
                continue

            try:
                values = list(map(int, line.split()))
                data_values.extend(values)
            except ValueError:
                print(f"Warning: Could not convert line to integers: '{line}'")

    # Convert data to the desired unit
    data_array = np.array(data_values)
    converted_data, unit_label = convert_data_to_unit(data_array, meas_tim, unit)

    return date_measured, meas_tim, converted_data


##  Parses the mca files
def read_measurement_data_mca(filename, unit):
    with open(filename, 'r') as file:
        lines = file.readlines()

    real_time = 0.0
    start_time = ""
    data_values = []
    data_section_started = False
    first_data_line = False

    for i, line in enumerate(lines):
        line = line.strip()

        # Extract real time and start time
        if line.startswith('REAL_TIME -') and i + 1 < len(lines):
            real_time = float(line.split('-')[-1].strip())

        elif line.startswith('START_TIME -') and i + 1 < len(lines):
            start_time = line.split('-')[-1].strip()
            start_time = parse_datetime_with_datetime_module(start_time)
        elif line == '<<DATA>>':
            data_section_started = True
            first_data_line = True
            continue

        # Process data section
        if data_section_started:
            if line.startswith('<<'):  # If new section starts, exit data reading
                break
            if first_data_line:
                first_data_line = False
                continue

            values = list(map(int, line.split()))
            data_values.extend(values)

    # Convert data to the unit
    data_array = np.array(data_values)
    converted_data, unit_label = convert_data_to_unit(data_array, real_time, unit)
    
    return real_time, start_time, converted_data


## Gets label molecule and aangle for each file
def extract_label_and_molecule(filename):
    parts = os.path.splitext(filename)[0].split('_')
    if len(parts) == 3:
        detector, molecule, angle = parts
        if molecule in ['AM', 'BA', 'CO', 'CS']:
            label = f"{detector} Detector, {molecule} Molecule, {angle} Degrees"
            return label, molecule, angle
    return None

## Processes Folder and returns a dictionary containing all the files data
def process_molecule_data(folder_paths, read_function, unit):
    molecule_data = {'AM': [], 'BA': [], 'CO': [], 'CS': []}

    for folder_path in folder_paths:
        for filename in os.listdir(folder_path):
            result = extract_label_and_molecule(filename)
            if result:
                label, molecule, angle = result
                # Read data using the provided reading function
                if read_function == read_measurement_data_mca:
                    real_time, start_time, data_array = read_measurement_data_mca(os.path.join(folder_path, filename), unit)
                    test_date = start_time
                    duration = real_time
                else:  # Assuming it's read_measurement_data_spe
                    date_measured, meas_tim, data_array = read_measurement_data_spe(os.path.join(folder_path, filename), unit)
                    test_date = date_measured
                    duration = meas_tim

                if data_array.size > 0:
                    molecule_data[molecule].append((data_array, label, test_date, duration))
                    
    # Process data into a dictionary
    result_dict = {}
    molecules = ['AM', 'BA', 'CO', 'CS']
    for molecule in molecules:
        datasets = molecule_data.get(molecule, [])
        molecule_dict = []

        if datasets:
            for data_array, label, test_date, duration in datasets:
                # Store data for returning
                molecule_dict.append({
                    'data_array': data_array,
                    'label': label,
                    'test_date': test_date,
                    'duration': duration
                })
        result_dict[molecule] = molecule_dict

    return result_dict

## Reading Files Associated functions
## converts the timestamps to UNIX time
def parse_datetime_with_datetime_module(time_str):
    # Parse the time string into a datetime object in UTC
    dt = datetime.strptime(time_str, "%m/%d/%Y %H:%M:%S").replace(tzinfo=timezone.utc)
    
    # Convert the datetime object to a UNIX timestamp
    timestamp = dt.timestamp()
    
    return timestamp

## Converts data array to specified unit 
def convert_data_to_unit(data_array, meas_tim, unit):
    """Converts the data array to the desired unit (CPS CPH or CPY)."""
    if unit == 'CPS':
        return data_array / meas_tim, 'Counts Per Second'
    elif unit == 'CPH':
        return (data_array / meas_tim) * 3600, 'Counts Per Hour'
    elif unit == 'CPY':
        return (data_array / meas_tim) * 3600*24*365, 'Counts Per Year'
    else:
        raise ValueError("Invalid unit. Please choose 'CPS' or 'CPH' or 'CPY'.") #Chat GPT Added the valueerror

def read_ranges_of_interest(roi_file):
    """Reads ranges of interest for each detector from a given text file."""
    ranges_of_interest = {}
    current_detector = None

    with open(roi_file, 'r') as file:
        for line in file:
            line = line.strip()
                
            if line.startswith('#'):  ## Start of a new detector section
                current_detector = line[1:]  ## Get the detector name without '#'
                ranges_of_interest[current_detector] = {}
                continue   
            if line:  # Process not empty lines
                molecule = line  # The current line is the molecule name
                line = next(file).strip()  # Get the next line for ranges
                ranges = list(map(int, line.split(',')))  # Convert to integer ranges
                ranges_of_interest[current_detector][molecule] = ranges  # Store in the dictionary

    return ranges_of_interest

def change_roi(detector):
    """Prompts the user to enter new ranges of interest for each detector."""
    new_ranges = {}

    print(f"\nEnter the ranges for {detector} (eg. Molecule, value1, value2..):")
    new_ranges[detector] = {}

        # Loop through each molecule
    for molecule in ['AM', 'BA', 'CO', 'CS']:
        ranges_input = input(f"{molecule}: ").strip()
        ranges = list(map(int, ranges_input.split(',')))
        new_ranges[detector][molecule] = ranges  

    return new_ranges



## Calibration Funcions

def create_masking_regions(ranges):
    """Creates masking regions by adding and subtracting 18 from each value in the ranges."""
    masking_regions = {}
    for molecule, values in ranges.items():
        masking_regions[molecule] = [(value - 18, value + 18) for value in values]
    return masking_regions

def plot_molecule_data(molecule_data, selected_angle, masking_regions):
    molecules = ['AM', 'BA', 'CO', 'CS']
    # Create a figure with 4 subplots
    fig, axes = plt.subplots(1, 4, figsize=(20, 6))
    fig.tight_layout(pad=4.0)

    for i, molecule in enumerate(molecules):
        ax = axes[i]
        datasets = molecule_data.get(molecule, [])
        
        # Check if there's any data for the molecule
        if not datasets:
            ax.set_title(f"{molecule} Molecule: No Data")
            ax.set_ylabel("Counts")
            continue
            
        for data in datasets:
            data_array = data['data_array']
            label = data['label']
            
            # Extract the exact angle from the label
            angle_in_label = label.split(", ")[2].replace(" Degrees", "").strip()  # Extract the angle

            # Check for an exact match
            if angle_in_label == selected_angle:
                # Plot each data array
                ax.plot(data_array, label=label)

                # Plot masking regions #chat gpt generated this
                if molecule in masking_regions:
                    for start, end in masking_regions[molecule]:
                        ax.axvspan(start, end, color='red', alpha=0.5, label='Masking Region')

        ax.set_title(f"{molecule} Molecule")
        ax.set_ylabel("Counts")
        ax.set_xlabel("Channel")
        ax.legend(loc='upper right')

    plt.show()

def plot_molecule_data_masked(molecule_data, selected_angle, masking_regions, line_type):
    molecules = ['AM', 'BA', 'CO', 'CS']
    
    # Initialize lists to store the initial guess parameters and Gaussian types
    all_initial_guess_params = []
    all_gauss_types = []
    energy_levels = []


    for molecule in molecules:
        datasets = molecule_data.get(molecule, [])

        for data in datasets:
            data_array = data['data_array']
            label = data['label']
            angle_in_label = label.split(", ")[2].replace(" Degrees", "").strip()

            if angle_in_label == selected_angle:
                has_data = True

                if molecule in masking_regions:
                    for i, (start, end) in enumerate(masking_regions[molecule]):
                        adjusted_start = max(0, start)

                        if adjusted_start < end and end <= len(data_array):
                            masked_segment = data_array[adjusted_start:end]
                            #channels = np.arange(len(masked_segment))
                            channels = np.arange(adjusted_start, end)

                            plt.figure(figsize=(10, 6))
                            plt.plot(channels, masked_segment, label=f'Masked Region {i + 1}', color='blue')
                            plt.title(f"{molecule} Molecule: Masked Region {i + 1} at {selected_angle} Degrees")
                            plt.ylabel("Counts")
                            plt.xlabel("Channels")
                            plt.legend(loc='upper right')
                            plt.grid()
                            plt.show()

                            # Collect initial guess parameters and Gaussian type
                            initial_guess_params, gauss_type, energy_level = ask_for_peak_info(masked_segment)
                            all_initial_guess_params.append(initial_guess_params)
                            all_gauss_types.append(gauss_type)
                            energy_levels.append(energy_level)

    return all_initial_guess_params, all_gauss_types, np.array(energy_levels)

def ask_for_peak_info(masked_data):
    # Calculate mean and standard deviation
    mean_value = np.mean(masked_data)
    std_value = np.std(masked_data)

    energy_level = float(input("What is the energy associated with this ROI"))
    gauss_type = input("Should a single or double Gaussian be fitted? (Enter 'single' or 'double'): ").strip().upper()
    
    if gauss_type == 'DOUBLE':
        A1 = float(input("Enter the amplitude of the first peak (A1): "))
        mu1 = float(input("Enter the corresponding channel of the first peak (A1): "))  
        std_dev1 = std_value  
        
        A2 = float(input("Enter the amplitude of the second peak (A2): "))
        mu2 = float(input("Enter the corresponding channel of the second peak (A2): "))   
        std_dev2 = std_value/2  
        
        return (mu1, std_dev1, A1, mu2, std_dev2, A2), gauss_type, energy_level
    
    elif gauss_type == 'SINGLE':
        A = float(input("Enter the amplitude of the peak (A): "))
        mu = float(input("Enter the corresponding channel of the first peak (A): "))
        std_dev = std_value / 2 
        
        return (mu, std_dev, A), gauss_type, energy_level
    
    else:
        print("Invalid input. Please enter 'SINGLE' or 'DOUBLE'.")
        return ask_for_peak_info(masked_data)

def get_molecule_data_masked(molecule_data, selected_angle, masking_regions):
    molecules = ['AM', 'BA', 'CO', 'CS']
    
    # Initialize a dictionary to store the masked data for each molecule
    masked_data = {molecule: [] for molecule in molecules}

    for molecule in molecules:
        datasets = molecule_data.get(molecule, [])

        if not datasets:
            print(f"{molecule} Molecule: No Data")
            continue

        for data in datasets:
            data_array = data['data_array']
            label = data['label']
            angle_in_label = label.split(", ")[2].replace(" Degrees", "").strip()

            if angle_in_label == selected_angle:
                if molecule in masking_regions:
                    for start, end in masking_regions[molecule]:
                        adjusted_start = max(0, start)

                        if adjusted_start < end and end <= len(data_array):
                            masked_segment = data_array[adjusted_start:end]
                            channels = np.arange(adjusted_start, end)
                            # Store the masked segment and channels for this molecule
                            masked_data[molecule].append((masked_segment, channels))

    return masked_data

## working best 
def plot_masked_regions_with_details(masked_data, initial_guess_params, gauss_types, lin_type, energy_levels):
    region_index = 0  ## Region count across all molecules
    max_values_resolution_and_channels = []
    
    for molecule, segments in masked_data.items():
        for i, (masked_segment, channels) in enumerate(segments):
            
            # Determine Gaussian type and prepare initial guess
            gauss_type = gauss_types[region_index]
            guess_params = initial_guess_params[region_index]
            energy_level = energy_levels[region_index]
            
            # Choose the fitting function
            if lin_type == 'LINEAR':
                fit_func = combined_linear if gauss_type == 'SINGLE' else combined_linear_double_gaussian
                initial_guess = [1, 1] + list(guess_params)  
            elif lin_type == 'QUADRATIC':
                fit_func = combined_quadratic if gauss_type == 'SINGLE' else combined_quadratic_double_gaussian
                initial_guess = [1, 1, 1] + list(guess_params)  
            elif lin_type == 'EXPONENTIAL':
                fit_func = combined_exponential if gauss_type == 'SINGLE' else combined_exponential_double_gaussian
                initial_guess = [1, 0.01] + list(guess_params)
            
            # Perform the curve fitting
            try: # Generated gpt due to constaint errors
                popt, _ = curve_fit(fit_func, channels, masked_segment, p0=initial_guess, maxfev=15000)
            except Exception as e:
                print(f"Fitting failed for {molecule} - Masked Region {region_index + 1}: {e}")
                continue

            # Create a figure for each masked region
            plt.figure(figsize=(8, 4))
            # Plot the data
            plt.plot(channels, masked_segment, 'o', label='Masked Data', color='blue')
            # Plot the fitted curve
            fitted_curve = fit_func(channels, *popt)
            plt.plot(channels, fitted_curve, '-', label='Fitted Curve', color='red')

            # Set plot details
            plt.title(f'{molecule} - Masked Region {region_index + 1}')
            plt.xlabel('Channel')
            plt.ylabel('Counts')
            plt.legend()
            plt.show()

            # Find the maximum value and corresponding channel in the fitted curve
            max_value = max(fitted_curve)
            max_channel = channels[list(fitted_curve).index(max_value)]
            
            # Print the parameters and Gaussian type below the plot
            print(f"Masked Region {region_index + 1}:\n")
            if gauss_type == 'SINGLE':
                mu, std_dev, A = guess_params
                FWHM = 2 * np.sqrt(2 * np.log(2)) * std_dev
                resolution = FWHM/energy_level
            
                print(f"  \nGaussian Type: {gauss_type}\n")
                print(f"  \nEnergy Resolution:{resolution}")
                print(f"  \nInitial Guess Parameters: Mu={mu}, Std Dev={std_dev}, Amplitude={A}")
                
            elif gauss_type == 'DOUBLE':
                mu1, std_dev1, A1, mu2, std_dev2, A2 = guess_params
                FWHM = 2 * np.sqrt(2 * np.log(2)) * std_dev1
                resolution = FWHM/energy_level
                
                print(f"  \nGaussian Type: {gauss_type}\n")
                print(f"  \nEnergy Resolution:{resolution}")                
                print(f"  \nInitial Guess Parameters:")
                print(f"    First Peak - Mu={mu1}, Std Dev={std_dev1}, Amplitude={A1}")
                print(f"    Second Peak - Mu={mu2}, Std Dev={std_dev2}, Amplitude={A2}\n")
            
            # Print optimized parameters from the fitting process
            print(f"\nOptimized Parameters:\n {popt}\n")
            print(f"  \nMaximum Value: {max_value} at Channel: {max_channel}\n")
            max_values_resolution_and_channels.append((max_value, max_channel, resolution))

            region_index += 1
    
    # Convert the list to a NumPy array and return
    return np.array(max_values_resolution_and_channels)

def plot_energy_levels(energy_level_data, energy_levels):
    
    energy_levels = np.array(energy_levels)
    # Extract max values and channels from the energy level data
    max_values = energy_level_data[:, 0]
    channels = energy_level_data[:, 1]
    resolution = energy_level_data[:,2]

    # Sort channels and corresponding energy levels
    sorted_indices = np.argsort(channels)
    sorted_channels = channels[sorted_indices]
    sorted_energy_levels = energy_levels[sorted_indices] 
    sorted_resolution = resolution[sorted_indices]
    
    plot_resolution_fit(energy_levels, resolution)
    
    # Create the figure
    plt.figure(figsize=(12, 6))
    plt.scatter(sorted_channels, sorted_energy_levels, color='red', marker='x', label='Energy Levels')    
    
    # Fit a linear line to the data
    linear_fit_params, _ = curve_fit(fit_linear, sorted_channels, sorted_energy_levels)
    fit_line = fit_linear(sorted_channels, *linear_fit_params)  # Calculate fitted values
    # Plot the fitted linear line
    plt.plot(sorted_channels, fit_line, color='blue', linestyle='--', label='Linear Fit')


        # Extract slope (a) and intercept (b)
    a, b = linear_fit_params
    # Display the equation of the line on the plot Chat GPT made this
    plt.text(.05, 0.95, f"y = {a:.2f}x + {b:.2f}", transform=plt.gca().transAxes, 
             fontsize=12, color='blue', verticalalignment='top')
    # Add labels and title
    plt.title("Energy Levels vs Channels with Linear Fit")
    plt.xlabel("Channel")
    plt.ylabel("Energy Level")
    plt.grid(True)
    plt.legend(loc='upper center')
    plt.tight_layout()
    plt.show()

    return linear_fit_params

def plot_resolution_fit(energy_levels, resolution):

    energy_levels = np.sort(energy_levels)
    sorted_indices = np.argsort(energy_levels)
    resolution = resolution[sorted_indices]

    # Calculate (δE)^2
    delta_E_squared = (resolution*energy_levels)**2

    # Log-transform the data
    log_energy_levels = np.log(energy_levels)
    log_delta_E_squared = np.log(delta_E_squared)

    # Fit the data to the quadratic model
    fit_params, pcov = curve_fit(quadratic_model, log_energy_levels, log_delta_E_squared)
    a, b, c = fit_params

    # Generate fitted values for (δE)^2 using the model
    fitted_delta_E_squared = quadratic_model(log_energy_levels, *fit_params)

    # Plot original data and fitted model
    plt.figure(figsize=(10, 6))
    plt.plot(np.exp(log_energy_levels), np.exp(log_delta_E_squared), 'o', label='Data (δE)^2')
    plt.plot(np.exp(log_energy_levels), np.exp(fitted_delta_E_squared), 'r-', label=f'Quadtric Fit')
    plt.xlabel("Energy Level (keV)")
    plt.ylabel("Resolution (δE)^2")
    plt.xscale("log")
    plt.yscale("log")
    
    plt.title("Resolution (δE)^2 vs Energy Levels with Quadratic Fit")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Print fit parameters for reference
    print(f"Fitted parameters:\na = {a:.2f}\nb = {b:.2f}\nc = {c:.2e}")
    return fit_params

## Off-Axis Response  
def off_axis_response(detector_data):
    molecule_data = {'AM': {}, 'BA': {}, 'CO': {}, 'CS': {}}
    
    for molecule, datasets in detector_data.items():
        for data_info in datasets:
            data_array, label = data_info['data_array'], data_info['label']
            _, _, angle = label.split(", ")
            angle = angle.split()[0]
            
            if data_array.size > 0:
                max_value = data_array.max()
                if angle not in molecule_data[molecule]:
                    molecule_data[molecule][angle] = []
                molecule_data[molecule][angle].append((max_value, label.split()[0]))

    # Plot the data
    fig, axs = plt.subplots(1, 4, figsize=(20, 5))
    molecules = ['AM', 'BA', 'CO', 'CS']
    for i, molecule in enumerate(molecules):
        ax = axs[i]
        angle_data = molecule_data.get(molecule, {})

        if angle_data:
            sorted_angles = sorted(angle_data.items(), key=lambda x: float(x[0])) # Chat gpt generated this

            for angle, max_values in sorted_angles:
                max_vals = [value for value, _ in max_values]
                detector_name = max_values[0][1] if max_values else 'Unknown Detector'
                ax.plot([float(angle)] * len(max_vals), max_vals, 'o', label=f'{detector_name} - Angle {angle}')

            ax.set_title(f'{molecule} ({detector_name})')
            ax.set_xlabel('Angles (Degrees)')
            if i == 0:
                ax.set_ylabel('Max Values')
            ax.grid(True)
            ax.legend()

    plt.tight_layout() 

    plt.show()


def perform_off_axis_response(cdte_folder, bgo_folder, nal_folder):
    
    # Prompt for unit input
    print("\nEnter what unit you would like the counts in:")
    print("'CPS' for Counts Per Second")
    print("'CPH' for Counts Per Hour")
    unit = input(" 'CPY' for Counts Per Year\n").strip().upper()
    
    # Process the data for each folder
    cdte_data = process_molecule_data([cdte_folder], read_measurement_data_mca, unit)
    bgo_data = process_molecule_data([bgo_folder], read_measurement_data_spe, unit)
    nal_data = process_molecule_data([nal_folder], read_measurement_data_spe, unit)
    
    # Plot Off Axis Response for each detector
    off_axis_response(cdte_data)
    off_axis_response(bgo_data)
    off_axis_response(nal_data)

## Reading Dictionary in
def parse_parameters_to_dict(filename):
    parameters = {
        "reference_date": None,
        "sources": [],
        "properties": {}
    }

    try:
        with open(filename, 'r') as file:
            lines = file.readlines()
    except FileNotFoundError:    # Chat gpt did errors
        print(f"Error: The file {filename} was not found.")
        return parameters 

    current_section = None
    line_iterator = iter(lines)

    for line in line_iterator:
        line = line.strip()

        if line.startswith("#Reference_date"):
            current_section = "reference_date"
            continue
        elif line.startswith("#Sources"):
            current_section = "sources"
            continue
        elif line.startswith("#Properties"):
            current_section = "properties"
            continue
        
        if current_section == "reference_date":
            if line:
                parameters["reference_date"] = datetime.strptime(line, '%m/%d/%Y %H:%M:%S')
                
        elif current_section == "sources":
            parts = line.split(", ")
            if len(parts) == 4:
                try:
                    source = {
                        "source": parts[0],
                        "source_no": parts[1],
                        "activity_uCi": float(parts[2]),
                        "accuracy": float(parts[3]),
                        "half_life": None  # Placeholder for half-life
                    }
                    parameters["sources"].append(source)
                except ValueError as e:    # chat gpt generated the error
                    print(f"Warning: Could not parse source line '{line}': {e}")
        
        elif current_section == "properties":
            if not line:
                continue
            
            parts = line.split(", ")
            if len(parts) >= 3:
                element = parts[0]
                decay_type = parts[1]
                number_of_energies = int(parts[2])
                
                parameters["properties"][element] = {
                    "decay_type": decay_type,
                    "half_life": None,
                    "half_life_uncertainty": None,
                    "energies": [],
                    "uncertainties_energies": [],
                    "intensities": [],
                    "uncertainties_intensities": []
                }

                # Read half-life and its uncertainty
                half_life_line = next(line_iterator, "").strip()
                if half_life_line:
                    half_life_parts = half_life_line.split(", ")
                    try:
                        parameters["properties"][element]["half_life"] = float(half_life_parts[0])  # Half-life
                        parameters["properties"][element]["half_life_uncertainty"] = float(half_life_parts[1])  # Half-life uncertainty
                        # Update the corresponding source with half-life
                        for source in parameters["sources"]:
                            if source["source"] == element:  # Assuming the element name matches source name
                                source["half_life"] = float(half_life_parts[0])
                                print(f"Assigned half-life {half_life_parts[0]} to source {element}")  # Debugging line
                    except ValueError as e:
                        print(f"Warning: Could not parse half-life line '{half_life_line}': {e}")

                # Read energies, uncertainties, intensities, etc.
                energies_line = next(line_iterator, "").strip()
                if energies_line:
                    parameters["properties"][element]["energies"] = list(map(float, energies_line.split(",")))

                uncertainties_energies_line = next(line_iterator, "").strip()
                if uncertainties_energies_line:
                    parameters["properties"][element]["uncertainties_energies"] = list(map(float, uncertainties_energies_line.split(",")))

                intensities_line = next(line_iterator, "").strip()
                if intensities_line:
                    parameters["properties"][element]["intensities"] = list(map(float, intensities_line.split(",")))

                uncertainties_intensities_line = next(line_iterator, "").strip()
                if uncertainties_intensities_line:
                    parameters["properties"][element]["uncertainties_intensities"] = list(map(float, uncertainties_intensities_line.split(",")))

    return parameters

## Activity Function 
def calculate_activity(parameters, source_name, time):
    
    # Convert the input time to a datetime object
    input_time = datetime.strptime(time, '%Y-%m-%d')
    
    # Find the source in the parameters
    for source in parameters["sources"]:
        if source["source"] == source_name:
            initial_activity = source["activity_uCi"]
            element = source_name.split("_")[0]

            #  half-life from the properties section
            half_life = parameters["properties"].get(element, {}).get("half_life")
            
            reference_date = parameters["reference_date"]
            if isinstance(reference_date, str):
                reference_date = datetime.strptime(reference_date, '%Y-%m-%d')
            
            # Calculate the time difference
            time_difference_years = (input_time - reference_date).days / 365  
            decay_constant = math.log(2) / half_life
            
            # Calculate the remaining activity using the decay formula:
            activity = initial_activity * math.exp(-decay_constant * time_difference_years)
            return activity
    
    print(f"Warning: Source '{source_name}' not found in parameters.")
    return None

## Efficiency 
def prompt_for_energy_and_emission_fraction(molecule, region):

    print(f"Enter data for {molecule} in region {region}:")
    energy = float(input(f"Enter the energy (in keV) for '{molecule}' in this region {region}: ").strip())
    emission_fraction = float(input(f"Enter the emission fraction for '{molecule}' at {energy} keV: ").strip())
    return energy, emission_fraction

def get_masked_regions_with_energy_and_emission(masking_regions):

    masked_regions_with_energy_and_emission = {}
    for molecule, regions in masking_regions.items():
        masked_regions_with_energy_and_emission[molecule] = []
        for region in regions:
            start, end = region
            if start < 0:
                print(f"Warning: Masking range ({start}, {end}) is out of bounds and will be skipped.")
                continue
            # Get energy and emission fraction from user
            energy, emission_fraction = prompt_for_energy_and_emission_fraction(molecule, region)
            # Append (region, energy, emission_fraction) to the list for this molecule
            masked_regions_with_energy_and_emission[molecule].append((region, energy, emission_fraction))
    return masked_regions_with_energy_and_emission

def calculate_activity_for_efficiency(parameters, element_name, time):

    source_mapping = {
        'AM': 'AM_241',
        'BA': 'BA_133',
        'CS': 'CS_137',
        'CO': 'CO_60'
    }

    element_key = element_name.split()[0].strip().upper()
    if isinstance(time, (float, int)):  
        input_time = datetime.fromtimestamp(time)
    elif isinstance(time, str):  
        input_time = datetime.strptime(time, '%Y-%m-%d')
    else:
        raise TypeError("Time must be a float (UNIX timestamp), int, or 'YYYY-MM-DD' string.")

    source_name = source_mapping.get(element_key)

    for source in parameters["sources"]:
        if source["source"] == source_name:
            initial_activity = source["activity_uCi"]
            half_life = parameters["properties"].get(element_key, {}).get("half_life")
            if half_life is None:
                print(f"Warning: Half-life for element '{element_key}' not specified.")
                return None

            reference_date = parameters["reference_date"]
            if isinstance(reference_date, str):
                reference_date = datetime.strptime(reference_date, '%Y-%m-%d')

            time_difference_years = (input_time - reference_date).days / 365.25
            decay_constant = math.log(2) / half_life
            activity = initial_activity * math.exp(-decay_constant * time_difference_years)
            return activity * 3.7e4  # Convert from µCi to Bq

    print(f"Warning: No source found for element '{element_key}'.")
    return None

def calculate_efficiency_data(detector_data, distance, parameters, masking_regions):

    efficiency_list = []  
    energy_cache = {}  # Chat gpt generated this cache idea into the code as original prompted every point
    radius = float(input('\nEnter the radius of the detector:\n'))
    
    for molecule, datasets in detector_data.items():
        if molecule not in masking_regions:
            print(f"Warning: No masking region specified for molecule '{molecule}'. Skipping.")
            continue

        for data_info in datasets:
            data_array, label, test_date = data_info['data_array'], data_info['label'], data_info['test_date']
            _, molecule_name, angle = label.split(", ")
            molecule_name = molecule_name.split()[0].strip().upper()  
            angle = angle.split()[0]

            for region in masking_regions[molecule]:
                roi_start, roi_end = region
                masked_data = data_array[roi_start:roi_end + 1]
                total_count_rate = np.sum(masked_data)

                # Check cache for energy and emission fraction
                if (molecule, region) not in energy_cache:
                    # Prompt the user for energy and emission fraction
                    energy = float(input(f"Enter the energy (in keV) for '{molecule}' in the region {roi_start}-{roi_end}: ").strip())
                    emission_fraction = float(input(f"Enter the emission fraction for '{molecule}' at {energy} keV: ").strip())
                    # Store in cache
                    energy_cache[(molecule, region)] = (energy, emission_fraction)
                else:
                    # Retrieve from cache
                    energy, emission_fraction = energy_cache[(molecule, region)]

                activity = calculate_activity_for_efficiency(parameters, molecule_name, test_date)
                effective_activity = activity * emission_fraction

                if effective_activity is None or effective_activity == 0:
                    print(f"Warning: Invalid effective activity for molecule '{molecule_name}' at energy {energy} keV.")
                    continue


                # Calculate the geometry factor using the solid angle formula for a circular detector
                solid_angle= (2* np.pi)*(1-(distance/np.sqrt(distance**2+radius**2)))
                # Calculate photon pass rate (photons per second reaching the detector)
                photon_pass_rate = (solid_angle/(4*np.pi)) * effective_activity
                # Calculate intrinsic efficiency
                intrinsic_efficiency = total_count_rate / photon_pass_rate
                # Calculate absolute efficiency
                absolute_efficiency = total_count_rate / effective_activity

                if isinstance(test_date, (float, int)):
                    test_date = datetime.fromtimestamp(test_date).strftime('%Y-%m-%d')

                efficiency_list.append({
                    'Molecule': molecule_name,
                    'Angle': angle,
                    'Energy (keV)': energy,
                    'Emission Fraction': emission_fraction,
                    'Intrinsic Efficiency': intrinsic_efficiency,
                    'Absolute Efficiency': absolute_efficiency,
                    'Activity': effective_activity,
                    'Total Count Rate': total_count_rate,
                    'Test Date': test_date
                })

    efficiency_data = pd.DataFrame(efficiency_list)
    plot_efficiencies_all_molecules(efficiency_data)
    return efficiency_data

def plot_efficiencies_all_molecules(efficiency_data):
    # Define selected angles for plotting
    selected_angles = [0, 45, 90]
    filtered_data = efficiency_data[(efficiency_data['Angle'].isin(selected_angles)) & 
                                    (efficiency_data['Intrinsic Efficiency']) & 
                                    (efficiency_data['Absolute Efficiency'])]

    # Set up subplots for intrinsic and absolute efficiencies
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True)
    
    intrinsic_fit_equations = []
    absolute_fit_equations = []
    
    for angle in selected_angles:
        angle_data = filtered_data[filtered_data['Angle'] == angle]
        sorted_indices = np.argsort(angle_data['Energy (keV)'].values)
        
        sorted_energy = angle_data['Energy (keV)'].values[sorted_indices]
        sorted_intrinsic_efficiency = angle_data['Intrinsic Efficiency'].values[sorted_indices]
        sorted_absolute_efficiency = angle_data['Absolute Efficiency'].values[sorted_indices]
        
        ln_energy = np.log(sorted_energy)
        
        # Fit for Intrinsic Efficiency chat gpt generated try except functions 
        try:
            popt_intrinsic, _ = curve_fit(efficiency_fit_function, ln_energy, np.log(sorted_intrinsic_efficiency))
            fitted_intrinsic = np.exp(efficiency_fit_function(ln_energy, *popt_intrinsic))
            axes[0].plot(sorted_energy, fitted_intrinsic, '--', color='gray')
            intrinsic_fit_equations.append(f"{angle}°: ln(ε) = {popt_intrinsic[0]:.2f} + {popt_intrinsic[1]:.2f} ln(E) + {popt_intrinsic[2]:.2f} (ln(E))^2")
        except Exception as e:
            print(f"Intrinsic efficiency fit failed for angle {angle}°: {e}")
        # Fit for Absolute Efficiency
        try:
            popt_absolute, _ = curve_fit(efficiency_fit_function, ln_energy, np.log(sorted_absolute_efficiency))
            fitted_absolute = np.exp(efficiency_fit_function(ln_energy, *popt_absolute))
            axes[1].plot(sorted_energy, fitted_absolute, '--', color='gray')
            absolute_fit_equations.append(f"{angle}°: ln(ε) = {popt_absolute[0]:.2f} + {popt_absolute[1]:.2f} ln(E) + {popt_absolute[2]:.2f} (ln(E))^2")
        except Exception as e:
            print(f"Absolute efficiency fit failed for angle {angle}°: {e}")

        axes[0].plot(sorted_energy, sorted_intrinsic_efficiency, 'o-', label=f"{angle}°")
        axes[1].plot(sorted_energy, sorted_absolute_efficiency, 'o-', label=f"{angle}°")
            
    # Format Intrinsic Efficiency plot
    axes[0].set_xscale("log")
    axes[0].set_xlabel("Energy (keV)")
    axes[0].set_ylabel("Intrinsic Efficiency")
    axes[0].set_title("Intrinsic Efficiency vs Energy at Selected Angles (0°, 45°, 90°)")
    axes[0].legend(title="Angle")
    axes[0].grid(True, which="both", linestyle="--", linewidth=0.5)

    # Format Absolute Efficiency plot
    axes[1].set_xscale("log")
    axes[1].set_xlabel("Energy (keV)")
    axes[1].set_ylabel("Absolute Efficiency")
    axes[1].set_title("Absolute Efficiency vs Energy at Selected Angles (0°, 45°, 90°)")
    axes[1].legend(title="Angle")
    axes[1].grid(True, which="both", linestyle="--", linewidth=0.5)

    # Display fit equations at the bottom of each plot
    axes[0].text(0, -0.3, "\n".join(intrinsic_fit_equations), transform=axes[0].transAxes, 
                 fontsize=9, verticalalignment='top', horizontalalignment='left')
    axes[1].text(0, -0.3, "\n".join(absolute_fit_equations), transform=axes[1].transAxes, 
                 fontsize=9, verticalalignment='top', horizontalalignment='left')

    plt.tight_layout()
    plt.show()

def preform_efficiency(cdte_folder, bgo_folder, nal_folder, parameters_data):

    # Define folder paths
    cdte_folder = 'CdTe1'  # CdTe folder path
    bgo_folder = 'BGO'      # BGO folder path
    nal_folder = 'Nal'      # Nal folder path
    parameters_file = 'Parameters.txt'  # parameters file path
    roi_file = 'ranges_of_interest.txt'
    ## Read Parameters and ROI text files
    parameters_data = parse_parameters_to_dict(parameters_file) 
    range_of_interest = read_ranges_of_interest(roi_file)

    ## Prompt for unit input
    print("\nEnter what unit you would like the counts in:")
    print("'CPS' for Counts Per Second")
    print("'CPH' for Counts Per Hour")
    unit = input(" 'CPY' for Counts Per Year\n").strip().upper()
            
    bgo_data = process_molecule_data([bgo_folder], read_measurement_data_spe, unit)
    nal_data = process_molecule_data([nal_folder], read_measurement_data_spe, unit)
    cdte_data = process_molecule_data([cdte_folder], read_measurement_data_mca, unit)
    
    cdte_ranges = range_of_interest.get('CDTE', {})
    cdte_masking_regions = create_masking_regions(cdte_ranges)
    bgo_ranges = range_of_interest.get('BGO', {})
    bgo_masking_regions = create_masking_regions(bgo_ranges)
    nal_ranges = range_of_interest.get('NAL', {})
    nal_masking_regions = create_masking_regions(nal_ranges)
    
    distance_bgo = float(input('\nEnter the distance between the detector and the source for the BGO detector:\n'))
    distance_nal = float(input('\nEnter the distance between the detector and the source for the Nal detector:\n'))    
    distance_cdte = float(input('\nEnter the distance between the detector and the source for the CdTe detector:\n'))    
    
    # Calculate efficiency data
    efficiency_table = calculate_efficiency_data(bgo_data, distance_bgo, parameters_data, bgo_masking_regions)
    efficiency_table1 = calculate_efficiency_data(nal_data, distance_nal, parameters_data, nal_masking_regions)
    efficiency_table2 = calculate_efficiency_data(cdte_data, distance_cdte, parameters_data, cdte_masking_regions)
        
    # Print the efficiency data table
    print(efficiency_table)
    print(efficiency_table1)
    print(efficiency_table2)

## Calibration
def perform_calibration(cdte_folder, bgo_folder, nal_folder, range_of_interest, parameters_data):
    
    ## BGO
    # Define folder paths
    cdte_folder = 'CdTe1'  # CdTe folder path
    bgo_folder = 'BGO'      # BGO folder path
    nal_folder = 'Nal'      # Nal folder path
    
    roi_file = 'ranges_of_interest.txt'  # Replace with the actual path to your file
    range_of_interest = read_ranges_of_interest(roi_file)

    
    # Prompt for unit input
    print("\nEnter what unit you would like the counts in:")
    print("'CPS' for Counts Per Second")
    print("'CPH' for Counts Per Hour")
    unit = input(" 'CPY' for Counts Per Year\n").strip().upper()
    
    selected_angle = input("\nAt what angle would you like to calibrate the detectors (recommended = 0 degrees)\n").strip().upper()

    detector = input("Which detector would you like to calibrate").strip().upper()

    if detector == "BGO":
        # Process the data for each folder
        bgo_data = process_molecule_data([bgo_folder], read_measurement_data_spe, unit)
        bgo_ranges = range_of_interest.get('BGO', {})
        bgo_masking_regions = create_masking_regions(bgo_ranges)
                
        # ROI confirmation 
        print("\nNow the Range of interest need to be confirmed.")
        print("The detector's data will be plotted followed by the pre-defined ROI for the molecules\n")
        
        plot_molecule_data(bgo_data, selected_angle, bgo_masking_regions)
        print("Ranges of Interest for BGO Detector:")
        print(bgo_ranges)
        
            # Ask if the user wants to change the ROI
        peaks = input("\nWould you like to continue with the default ROI for the detectors (recommended)\n").strip().upper()
            
        if peaks == 'NO':
            updated_ranges = change_roi('BGO')
            print("\nUpdated ranges of interest:", updated_ranges)
            bgo_ranges = updated_ranges.get('BGO', {})
        
        bgo_masking_regions = create_masking_regions(bgo_ranges)
            
        print("\nConfirmed ROIs\n")
        print("\nMasking Regions for BGO Detector:")
        print(bgo_masking_regions)
        
        line_type = input("\nWould you like to fit a linear, quadratic, or exponential line to the data?\n").strip().upper()
        
        initial_guess_params_bgo, gauss_type_bgo, energy_levels_bgo = plot_molecule_data_masked(bgo_data, selected_angle, bgo_masking_regions, line_type)
        print(energy_levels_bgo)
        
        masked_data_bgo = get_molecule_data_masked(bgo_data, selected_angle, bgo_masking_regions,)
        
        energy_level_data_bgo = plot_masked_regions_with_details(masked_data_bgo, initial_guess_params_bgo, gauss_type_bgo, line_type, energy_levels_bgo)
        linear_fit_params_bgo,b = plot_energy_levels(energy_level_data_bgo, energy_levels_bgo)

    if detector == "NAL":
        # Process the data for each folder
        nal_data = process_molecule_data([nal_folder], read_measurement_data_spe, unit)
        
        nal_ranges = range_of_interest.get('NAL', {})
        nal_masking_regions = create_masking_regions(nal_ranges)
                
        # ROI confirmation 
        print("\nNow the Range of interest need to be confirmed.")
        print("The detector's data will be plotted followed by the pre-defined ROI for the molecules\n")
        
        plot_molecule_data(nal_data, selected_angle, nal_masking_regions)
        print("Ranges of Interest for NAL Detector:")
        print(nal_ranges)
        
            # Ask if the user wants to change the ROI
        peaks = input("\nWould you like to continue with the default ROI for the detectors (recommended)\n").strip().upper()
            
        if peaks == 'NO':
            updated_ranges = change_roi('NAL')
            print("\nUpdated ranges of interest:", updated_ranges)
            nal_ranges = updated_ranges.get('NAL', {})
        
        nal_masking_regions = create_masking_regions(nal_ranges)
            
        print("\nConfirmed ROIs\n")
        print("\nMasking Regions for NAL Detector:")
        print(nal_masking_regions)
        
        line_type = input("\nWould you like to fit a linear, quadratic, or exponential line to the data?\n").strip().upper()
        
        initial_guess_params_nal, gauss_type_nal, energy_levels_nal = plot_molecule_data_masked(nal_data, selected_angle, nal_masking_regions, line_type)
        print(energy_levels_nal)
        
        masked_data_nal = get_molecule_data_masked(nal_data, selected_angle, nal_masking_regions,)
        
        energy_level_data_nal = plot_masked_regions_with_details(masked_data_nal, initial_guess_params_nal, gauss_type_nal, line_type, energy_levels_nal)
        linear_fit_params_nal,b = plot_energy_levels(energy_level_data_nal, energy_levels_nal)

    if detector == "CDTE":
        # Process the data for each folder
        cdte_data = process_molecule_data([cdte_folder], read_measurement_data_mca, unit)        
        cdte_ranges = range_of_interest.get('CDTE', {})
        cdte_masking_regions = create_masking_regions(cdte_ranges)
                
        # ROI confirmation 
        print("\nNow the Range of interest need to be confirmed.")
        print("The detector's data will be plotted followed by the pre-defined ROI for the molecules\n")
        
        plot_molecule_data(cdte_data, selected_angle, cdte_masking_regions)
        print("Ranges of Interest for CdTe Detector:")
        print(cdte_ranges)
        
            # Ask if the user wants to change the ROI
        peaks = input("\nWould you like to continue with the default ROI for the detectors (recommended)\n").strip().upper()
            
        if peaks == 'NO':
            updated_ranges = change_roi('CDTE')
            print("\nUpdated ranges of interest:", updated_ranges)
            cdte_ranges = updated_ranges.get('CDTE', {})
        
        cdte_masking_regions = create_masking_regions(cdte_ranges)
            
        print("\nConfirmed ROIs\n")
        print("\nMasking Regions for CdTe Detector:")
        print(cdte_masking_regions)
        
        line_type = input("\nWould you like to fit a linear, quadratic, or exponential line to the data?\n").strip().upper()
        
        initial_guess_params_cdte, gauss_type_cdte, energy_levels_cdte = plot_molecule_data_masked(cdte_data, selected_angle, cdte_masking_regions, line_type)
        print(energy_levels_cdte)
        
        masked_data_cdte = get_molecule_data_masked(cdte_data, selected_angle, cdte_masking_regions,)
        
        energy_level_data_cdte = plot_masked_regions_with_details(masked_data_cdte, initial_guess_params_cdte, gauss_type_cdte, line_type, energy_levels_cdte)
        linear_fit_params_cdte,b = plot_energy_levels(energy_level_data_cdte, energy_levels_cdte)


def main_menu():

    # Define folder paths
    cdte_folder = 'CdTe1'  # CdTe folder path
    bgo_folder = 'BGO'      # BGO folder path
    nal_folder = 'Nal'      # Nal folder path
    roi_file = 'ranges_of_interest.txt'  # Replace with the actual path to your file
    range_of_interest = read_ranges_of_interest(roi_file)
        
    filename = 'Parameters.txt'  # Replace with your actual file path
    parameters_data = parse_parameters_to_dict(filename)

    while True:
        print("\nMain Menu:")
        print("1. Calibration")
        print("2. Calculate Activity")
        print("3. Off-Axis Response")
        print("4. Calculate Efficiency")
        print("5. Exit")
        
        choice = input("Select an option (1-5): ")

        if choice == '1':
            perform_calibration(cdte_folder, bgo_folder, nal_folder, range_of_interest, parameters_data)
        elif choice == '2':
            date = input("\nEnter the time you want the activity for (YYYY-MM-DD)\n")
            print("Enter the molecule you want the activity for. Options:\n")
            print(" AM_241")
            print(" BA_133")
            print(" CS_137")
            molecule = input(" CO_60\n")
            
            activity_at_time = calculate_activity(parameters_data, molecule, date)
            print(f"\nActivity at specified time: {activity_at_time} µCi\n")
        elif choice == '3':
            perform_off_axis_response(cdte_folder, bgo_folder, nal_folder)
        elif choice == '4':
            preform_efficiency(cdte_folder, bgo_folder, nal_folder, parameters_data)
        elif choice == '5':
            print("Exiting program.")
            break
        else:
            print("Invalid selection. Please try again.")

if __name__ == "__main__":
    main_menu()
