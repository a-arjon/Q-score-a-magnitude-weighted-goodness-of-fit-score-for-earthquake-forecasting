import os
import pandas as pd
import h5py
import numpy as np
import calendar
import re
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import itertools
from itertools import cycle

import csep
from csep.core import poisson_evaluations as poisson
from scipy.stats import poisson as poisson_stat
from joblib import Parallel, delayed


def load_hdf5_to_dataframe_year_month(hdf5_filename, year_to_imp, month_to_imp):
    """
    Loads data from an HDF5 file into a nested dictionary of pandas DataFrames 
    for a specified year and month.

    Args:
        hdf5_filename (str): Path to the HDF5 file.
        year_to_imp (str): Year to import (e.g., '2022').
        month_to_imp (str): Month to import (e.g., '01' for January).

    Returns:
        dict: Nested dictionary of DataFrames organized as:
              {
                  'year': {
                      'month': {
                          'filename': pd.DataFrame
                      }
                  }
              }
    """
    # Initialize an empty dictionary to store the DataFrames
    dataframes = {}
    
    # Open the HDF5 file in read mode
    with h5py.File(hdf5_filename, 'r') as hdf5_file:
        # Iterate through each year in the HDF5 file
        for year in hdf5_file:
            # Check if the current year matches the specified year
            if year == year_to_imp:
                print(year)  # Debug: Print the year being processed
                
                # Access the group corresponding to the current year
                year_d = hdf5_file[year]
                dataframes[year] = {}  # Initialize nested dict for the year
                
                # Iterate through each month in the current year
                for month in year_d:
                    # Check if the current month matches the specified month
                    if month == month_to_imp:
                        dataframes[year][month] = {}  # Initialize nested dict for the month
                        
                        # Access the group corresponding to the current month
                        month_d = year_d[month]
                        
                        # Iterate through each file in the current month
                        for file_ in month_d:
                            # Load the file's data into a DataFrame with float32 dtype
                            df = pd.DataFrame(month_d[file_][:], dtype=np.float32)
                            
                            # Store the DataFrame in the dictionary
                            dataframes[year][month][file_] = df
    
    # Return the nested dictionary of DataFrames
    return dataframes

def extract_file_date(filename):
    """
    Extracts the date from a filename in the format d_m_yyyy or dd_mm_yyyy.

    Args:
        filename (str): The name of the file from which to extract the date.

    Returns:
        str or None: The extracted date as a string if found (e.g., '12_01_2022'), 
                     otherwise None.
    """
    # Regular expression pattern to match the date in the format d_m_yyyy or dd_mm_yyyy
    date_pattern = re.compile(r'\d{1,2}_\d{1,2}_\d{4}')
    
    # Search for the pattern in the filename
    match = date_pattern.search(filename)
    
    # If a match is found, return the matched date string
    if match:
        return match.group()
    else:
        # Return None if no date is found in the filename
        return None
    
def extract_all_dates(fore_dict):
    """
    Extracts all forecast dates from a nested dictionary of file names.

    Args:
        fore_dict (dict): A nested dictionary organized as:
                          {
                              'year': {
                                  'month': {
                                      'filename': DataFrame
                                  }
                              }
                          }

    Returns:
        list: A list of datetime objects representing all extracted dates.
    """
    # Initialize an empty list to store the extracted dates
    fore_date_list = []
    
    # Iterate through each year in the dictionary
    for year in fore_dict.keys():
        year_d = fore_dict[year]
        
        # Iterate through each month in the current year
        for month in year_d.keys():
            month_d = year_d[month]
            
            # Iterate through each file name in the current month
            for file_name in month_d.keys():
                # Extract the date string from the file name
                file_date_string = extract_file_date(file_name)
                
                # Convert the date string to a datetime object
                file_date = datetime.strptime(file_date_string, '%m_%d_%Y')
                
                # Append the datetime object to the list
                fore_date_list.append(file_date)
    
    # Return the list of extracted dates
    return fore_date_list

def load_hdf5_to_dict(hdf5_filename, start_date='', end_date='', all_dates=False):
    """
    Loads data from an HDF5 file into a nested dictionary of pandas DataFrames. 
    The data can be filtered by a date range or loaded entirely.

    Args:
        hdf5_filename (str): Path to the HDF5 file.
        start_date (str, optional): Start date in the format 'dd/mm/YYYY'. Required if all_dates is False.
        end_date (str, optional): End date in the format 'dd/mm/YYYY'. Required if all_dates is False.
        all_dates (bool, optional): If True, loads all available dates. 
                                    If False, loads data within the specified date range.

    Returns:
        dict: Nested dictionary of DataFrames organized as:
              {
                  'year': {
                      'month': {
                          'filename': pd.DataFrame
                      }
                  }
              }
    """
    # Initialize an empty dictionary to store the DataFrames
    dataframes = {}
    
    # Open the HDF5 file in read mode
    with h5py.File(hdf5_filename, 'r') as hdf5_file:
        # If all_dates is True, load the entire dataset
        if all_dates:
            # Iterate through each year in the HDF5 file
            for year in hdf5_file:
                print('Loading year:', year)  # Debug: Print the year being loaded
                
                # Access the group corresponding to the current year
                year_d = hdf5_file[year]
                dataframes[year] = {}  # Initialize nested dict for the year
                
                # Iterate through each month in the current year
                for month in year_d:
                    dataframes[year][month] = {}  # Initialize nested dict for the month
                    
                    # Access the group corresponding to the current month
                    month_d = year_d[month]
                    
                    # Iterate through each file in the current month
                    for file_ in month_d:
                        # Load the file's data into a DataFrame with float32 dtype
                        df = pd.DataFrame(month_d[file_][:], dtype=np.float32)
                        
                        # Store the DataFrame in the dictionary
                        dataframes[year][month][file_] = df
        
        # If all_dates is False, filter data within the specified date range
        else:
            # Convert start and end dates from strings to datetime objects
            start_date = datetime.strptime(start_date, '%d/%m/%Y')
            end_date = datetime.strptime(end_date, '%d/%m/%Y')
            
            # Iterate through each year in the HDF5 file
            for year in hdf5_file:
                # Check if the current year is within the date range
                if int(year) >= int(start_date.year) and int(year) <= int(end_date.year):
                    print('Loading year:', year)  # Debug: Print the year being loaded
                    
                    # Access the group corresponding to the current year
                    year_d = hdf5_file[year]
                    dataframes[year] = {}  # Initialize nested dict for the year
                    
                    # Iterate through each month in the current year
                    for month in year_d:
                        print('Loading month:', month)  # Debug: Print the month being loaded
                        
                        # Construct datetime objects for date comparison
                        fore_part_date = datetime(int(year), int(month), 1)
                        start_part_date = datetime(start_date.year, start_date.month, 1)
                        last_day = calendar.monthrange(end_date.year, end_date.month)[1]
                        end_part_date = datetime(end_date.year, end_date.month, last_day)
                        
                        # Check if the month is within the date range
                        if fore_part_date >= start_part_date and fore_part_date <= end_part_date:
                            dataframes[year][month] = {}  # Initialize nested dict for the month
                            
                            # Access the group corresponding to the current month
                            month_d = year_d[month]
                            
                            # Iterate through each file in the current month
                            for file_ in month_d:
                                # Extract the date from the file name
                                file_date_str = extract_file_date(file_)
                                
                                # Convert the extracted date string to a datetime object
                                file_date = datetime.strptime(file_date_str, '%m_%d_%Y')
                                
                                # Check if the file date is within the date range
                                if file_date >= start_date and file_date <= end_date:
                                    # Load the file's data into a DataFrame with float32 dtype
                                    df = pd.DataFrame(month_d[file_][:], dtype=np.float32)
                                    
                                    # Store the DataFrame in the dictionary
                                    dataframes[year][month][file_] = df
    
    # Return the nested dictionary of DataFrames
    return dataframes

def get_cumulative_forecast(dict_fore, forecast_name, start_date='', end_date='', all_dates=False, waterlevel = 0):
    """
    Creates a cumulative forecast from a nested dictionary of pandas DataFrames 
    between a start and end date.

    Args:
        dict_fore (dict): Nested dictionary of DataFrames organized as:
                          {
                              'year': {
                                  'month': {
                                      'filename': pd.DataFrame
                                  }
                              }
                          }
        forecast_name (str): Name for the cumulative forecast.
        start_date (str, optional): Start date in the format 'dd/mm/YYYY HH:MM:SS'. 
                                    Required if all_dates is False.
        end_date (str, optional): End date in the format 'dd/mm/YYYY HH:MM:SS'. 
                                  Required if all_dates is False.
        all_dates (bool, optional): If True, processes all available dates. 
                                    If False, processes data within the specified date range.

    Returns:
        csep.Forecast: Cumulative forecast object loaded using csep.
    """
    # Initialize an array to accumulate the forecast data
    # First find number of rows
    dd = dict_fore
    while isinstance(dd, dict):
        dd = next(iter(dd.values()))
    n_row = dd.shape[0]
    del dd
    N_total_per_bin = np.repeat(0, n_row)
    
    # If all_dates is True, accumulate forecasts for all dates
    if all_dates:
        # Iterate through each year in the dictionary
        for year in dict_fore.keys():
            # Iterate through each month in the current year
            for month in dict_fore[year].keys():
                # Iterate through each forecast file in the current month
                for fore in dict_fore[year][month].keys():
                    # Copy the DataFrame to avoid modifying the original
                    tmp = dict_fore[year][month][fore].copy()
                    
                    # Accumulate the forecast data
                    if len(tmp[8]) != n_row:
                        zeros_t = pd.Series([0] * abs(n_row - len(tmp[8])))
                        tmp_z = pd.concat([tmp[8].fillna(0), zeros_t], ignore_index=True)
                        N_total_per_bin += tmp_z
                    else:  
                        N_total_per_bin += tmp[8]
                    #Accumulate the forecast data
                    #N_total_per_bin += tmp[8]
    
    # If all_dates is False, filter data within the specified date range
    else:
        # Convert start and end dates from strings to datetime objects
        start_date = datetime.strptime(start_date, '%d/%m/%Y %H:%M:%S')
        end_date = datetime.strptime(end_date, '%d/%m/%Y %H:%M:%S')
        
        # Iterate through each year in the dictionary
        for year in dict_fore.keys():
            # Check if the current year is within the date range
            if int(year) >= int(start_date.year) and int(year) <= int(end_date.year):
                
                # Iterate through each month in the current year
                for month in dict_fore[year].keys():
                    # Construct datetime objects for date comparison
                    fore_part_date = datetime.strptime(month + '/' + year, '%m/%Y')
                    start_part_date = datetime.strptime(str(start_date.month) + '/' + str(start_date.year), '%m/%Y')
                    end_part_date = start_part_date + timedelta(days=30)
                    
                    # Check if the month is within the date range
                    if fore_part_date >= start_part_date and fore_part_date <= end_part_date:
                        
                        # Iterate through each file in the current month
                        for file_ in dict_fore[year][month].keys():
                            # Extract the date from the file name
                            file_date_str = extract_file_date(file_)
                            
                            # Convert the extracted date string to a datetime object
                            file_date = datetime.strptime(file_date_str, '%m_%d_%Y')
                            
                            # Check if the file date is within the date range
                            if file_date >= start_date and file_date <= end_date:
                                # Copy the DataFrame to avoid modifying the original
                                tmp = dict_fore[year][month][file_].copy()
                                
                                # Accumulate the forecast data
                                N_total_per_bin = N_total_per_bin + tmp[8]
    
    
    # set zero bins to waterlevel
    target_idx = N_total_per_bin == 0
    N_total_per_bin[target_idx] = waterlevel
    # Update the DataFrame with the cumulative forecast data
    tmp[8] = N_total_per_bin
    tmp = tmp.iloc[:,0:10]
   
    # Save the cumulative forecast to a temporary file
    tmp.to_csv('tmp.dat', sep='\t', header=None, index=None)
    forecast_ = csep.load_gridded_forecast('tmp.dat', name=forecast_name)
    os.remove('tmp.dat')
    # Load the cumulative forecast using csep and return the result
    return forecast_




def create_hdf5_from_folder_structure(root_dir, hdf5_filename, date_column='date', chunks_flag=False, 
                                      data_type='num', compression_n=9, sep_file='\t'):
    """
    Create an HDF5 file from a directory structure where data is organized into year/month folders.

    This function traverses a directory structure organized by year and month, reads `.dat` files, and 
    stores the data in an HDF5 file. Each dataset in the HDF5 file corresponds to a `.dat` file, with the 
    path structure reflecting the year, month, and model. The data is compressed using gzip with a user-specified 
    compression level.

    Args:
        root_dir (str): The root directory containing the year/month folders with the `.dat` files.
        hdf5_filename (str): The name of the output HDF5 file.
        date_column (str, optional): The name of the column representing the date in the data file (default is 'date').
        chunks_flag (bool, optional): If True, creates datasets with chunking enabled in the HDF5 file (default is False).
        data_type (str, optional): The type of the data, either 'num' for numerical or 'str' for string data (default is 'num').
        compression_n (int, optional): The level of compression (1-9) for the gzip compression (default is 9).
        sep_file (str, optional): The separator used in the `.dat` files (default is '\t').

    Returns:
        None: The function creates the HDF5 file and prints a success message.
    
    Raises:
        FileNotFoundError: If the specified root directory or any of the files cannot be found.
        ValueError: If an invalid data type is specified.
    """
    # Create an HDF5 file
    with h5py.File(hdf5_filename, 'w') as hdf5_file:
        # Loop through the directory structure
        for year in os.listdir(root_dir):
            year_path = os.path.join(root_dir, year)
            if os.path.isdir(year_path):
                for month in os.listdir(year_path):
                    month_path = os.path.join(year_path, month)
                    if os.path.isdir(month_path):
                        for file in os.listdir(month_path):
                            if file.endswith('.dat'):
                                # Extract date and model from the file name
                                date, model = file.split('.')[0], file.split('.')[1]

                                # Read the CSV file into a DataFrame
                                file_path = os.path.join(month_path, file)
                                df = pd.read_csv(file_path, sep=sep_file, header=None)
                                
                                # Handle potential separator issues
                                if df.shape[1] == 1:
                                    df = pd.read_csv(file_path, sep='\t', header=None)

                                # Process the data based on the data type
                                if data_type == 'num':
                                    # Replace non-standard minus signs with a valid minus sign
                                    df = df.replace('–', '-', regex=True)
                                    data = df.values.astype(np.float32)  # Convert to float32 for numerical data
                                elif data_type == 'str':
                                    data = df.astype(str).to_numpy()  # Convert to string data if specified
                                else:
                                    raise ValueError(f"Invalid data_type '{data_type}' specified. Must be 'num' or 'str'.")
                                
                                # Create a dataset path in the HDF5 file
                                dataset_path = f"/{year}/{month}/{date}_{model}"
                                
                                # Create the dataset in the HDF5 file with or without chunking
                                if chunks_flag:
                                    hdf5_file.create_dataset(dataset_path, data=data, compression="gzip", 
                                                              compression_opts=compression_n, chunks=True)
                                else:
                                    hdf5_file.create_dataset(dataset_path, data=data, compression="gzip", 
                                                              compression_opts=compression_n)
                                
                                # Save column names as an attribute of the dataset
                                hdf5_file[dataset_path].attrs['columns'] = df.columns.tolist()

    print(f"HDF5 file '{hdf5_filename}' created successfully.")


def print_tree(startpath, indent_level=0):
    """
    Recursively prints the directory structure of a given path in a tree-like format.

    This function walks through the directory structure starting from `startpath` and prints 
    out each directory and its contents in a hierarchical, indented format. It visually represents
    the folder structure with directories and files using `|--` as markers.

    Args:
        startpath (str): The root directory where the traversal begins.
        indent_level (int, optional): The initial indentation level. Default is 0. 

    Returns:
        None: The function prints the directory structure to the console.

    Example:
        print_tree('/home/user/Documents')
        This will print the directory structure starting from '/home/user/Documents'.
    """
    
    # Walk through the directory structure
    for root, dirs, files in os.walk(startpath):
        # Calculate the level of indentation based on the depth in the directory tree
        level = root.replace(startpath, '').count(os.sep)
        indent = ' ' * 4 * (level)  # Indentation for directories
        print(f"{indent}|-- {os.path.basename(root)}/")  # Print the directory name

        # Indentation for files inside the directory
        sub_indent = ' ' * 4 * (level + 1)
        
        # Loop through the files and print them with appropriate indentation
        for f in files:
            print(f"{sub_indent}|-- {f}")
