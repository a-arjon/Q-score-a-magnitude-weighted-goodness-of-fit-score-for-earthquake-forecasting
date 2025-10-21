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

def get_mag_plot(forecasts_list, catalog, save_fig=False, fig_name='fig.png', fig_dpi=100):
    """
    Plots the magnitude distribution of multiple forecasts compared to observed earthquake magnitudes.

    Args:
        forecasts_list (list): List of forecast objects containing magnitude information.
        catalog (object): Catalog object containing observed earthquake magnitudes.
        save_fig (bool, optional): If True, saves the figure as an image file. Default is False.
        fig_name (str, optional): Filename for saving the figure. Default is 'a.png'.
        fig_dpi (int, optional): Resolution (DPI) for the saved figure. Default is 100.

    Returns:
        None
    """
    # Initialize an iterator for colors using the tab10 colormap
    colors = itertools.cycle(plt.cm.tab10.colors)
    
    # Create a new figure
    plt.figure()
    
    # Iterate through each forecast in the list
    for forecast in forecasts_list:
        # Get the next color for the current forecast
        color = next(colors)
        
        # Get the magnitude counts and values from the forecast
        mag_count = forecast.magnitude_counts()
        mag_val = forecast.get_magnitudes()
        
        # Filter out zero counts for better visualization
        mag_val = [mag_val[idx1] for idx1 in np.arange(1, len(mag_count)) if mag_count[idx1] != 0]
        mag_count = [mag_count[idx2] for idx2 in np.arange(1, len(mag_count)) if mag_count[idx2] != 0]
        
        # Get the observed magnitudes from the catalog
        mag_obs = catalog.get_magnitudes()
        
        # Count observed magnitudes in the same bins as the forecast
        mag_obs_count = [
            np.sum([mag <= mag_o < (mag + 0.1) for mag_o in mag_obs]) 
            for mag in mag_val
        ]
        
        # Filter out zero counts for observed magnitudes
        mag_val2 = [mag_val[idx3] for idx3 in np.arange(0, len(mag_obs_count)) if mag_obs_count[idx3] != 0]
        mag_obs_count = [m for m in mag_obs_count if m != 0]
        
        # Plot forecast magnitude distribution
        plt.scatter(mag_val, np.log10(mag_count), color=color, label=forecast.name)
    
    # Plot observed magnitude distribution
    plt.scatter(
        mag_val2, 
        np.log10(mag_obs_count), 
        color='black', 
        marker='x', 
        label='$M \\geq m$ earthquakes'
    )
    
    # Set the y-axis label to log scale for earthquake counts
    plt.ylabel('log10(Earthquakes with $M \\geq m$)')
    
    # Set the x-axis label for magnitudes
    plt.xlabel('Magnitude ($m$)')
    
    # Add a legend with a transparent frame
    plt.legend(
        frameon=True, 
        fancybox=True, 
        framealpha=0.2, 
        facecolor='lightblue', 
        edgecolor='grey',
        fontsize=12
    ).get_frame().set_linewidth(2)
    
    # Save the figure if requested
    if save_fig:
        plt.savefig(fig_name, dpi=fig_dpi, bbox_inches='tight')
    
    # Display the plot
    plt.show()

def get_N_per_fore(dict_fore):
    """
    Calculate the number of events per forecast and return a sorted DataFrame.

    This function extracts the number of events from the forecast data, which is organized by
    year, month, and forecast. For each forecast, it computes the total number of events and
    the corresponding date (as a datetime object). It then returns a sorted DataFrame with the
    cumulative sum of the events.

    Args:
        dict_fore (dict): Dictionary containing forecast data organized by year, month, and forecast.
                          Each forecast contains an array where the 8th index is the number of events.

    Returns:
        pd.DataFrame: A DataFrame with the number of events (`N`) and corresponding `day` (datetime).
                       The DataFrame is sorted by `day` and includes the cumulative sum (`cumsum`) of events.
    """
    # Lists to store the number of events and the corresponding forecast date
    N_per_fore = []
    day_fore = []

    # Loop through each year in the dictionary
    for year in dict_fore.keys():
        # Loop through each month within the year
        for month in dict_fore[year].keys():
            # Loop through each forecast in the month
            for fore in dict_fore[year][month].keys():
                # Append the total number of events (from index 8 of forecast data) to the list
                N_per_fore.append(np.sum(dict_fore[year][month][fore][8]))
                # Append the corresponding date of the forecast (extracted from the forecast file name)
                day_fore.append(datetime.strptime(extract_file_date(fore), '%m_%d_%Y'))

    # Create a DataFrame with 'N' (number of events) and 'day' (date)
    df_out = pd.DataFrame({'N': N_per_fore, 'day': day_fore}, index=None)

    # Sort the DataFrame by 'day' to ensure it is ordered chronologically
    df_sorted = df_out.sort_values(by='day')

    # Compute the cumulative sum of the events and add it to the DataFrame
    df_sorted['cumsum'] = df_sorted['N'].cumsum()

    return df_sorted  # Return the sorted DataFrame with cumulative sum


def get_N_CI(df_N, alpha_level):
    """
    Calculate the Poisson confidence intervals for the number of events and their cumulative sum.

    This function computes the lower and upper bounds of the Poisson confidence intervals for both
    the number of events per forecast and their cumulative sum. The confidence intervals are calculated
    using the given alpha level, which represents the significance level (usually 0.05 for a 95% confidence interval).

    Args:
        df_N (pd.DataFrame): DataFrame containing the number of events and cumulative sum.
                              The DataFrame should have columns 'N' for the number of events and 'cumsum' for the cumulative sum.
        alpha_level (float): Significance level for the confidence interval. A typical value is 0.05, which corresponds to a 95% confidence interval.

    Returns:
        pd.DataFrame: A DataFrame containing the lower and upper bounds of the Poisson confidence intervals
                      for both the daily number of events and the cumulative number of events. The resulting
                      DataFrame contains the following columns:
                      - 'N_lower_bound': Lower bound of the Poisson confidence interval for the daily number of events.
                      - 'N_upper_bound': Upper bound of the Poisson confidence interval for the daily number of events.
                      - 'N_cs_lower_bound': Lower bound of the Poisson confidence interval for the cumulative number of events.
                      - 'N_cs_upper_bound': Upper bound of the Poisson confidence interval for the cumulative number of events.
    """
    # Calculate the lower and upper Poisson confidence intervals for the daily number of events
    N_CI_low = [poisson_stat.ppf(alpha_level/2, N) for N in df_N['N']]
    N_CI_up = [poisson_stat.ppf(1 - alpha_level/2, N) for N in df_N['N']]

    # Calculate the lower and upper Poisson confidence intervals for the cumulative number of events
    N_CI_low_cs = [poisson_stat.ppf(alpha_level/2, N) for N in df_N['cumsum']]
    N_CI_up_cs = [poisson_stat.ppf(1 - alpha_level/2, N) for N in df_N['cumsum']]

    # Return a DataFrame with the Poisson confidence intervals
    return pd.DataFrame({
        'N_lower_bound': N_CI_low,
        'N_upper_bound': N_CI_up,
        'N_cs_lower_bound': N_CI_low_cs,
        'N_cs_upper_bound': N_CI_up_cs,
    })

def get_N_plot(dict_fore, catalog, alpha_level = 0.05, type = 'cumulative', size_p = 10, forecast_label = 'forecast',
               return_df = False, show_obs_counts = False):
    """
    Plot the evolution of the number of events per day, either in cumulative or rate form.

    This function generates a plot showing the evolution of forecasted earthquake counts over time
    and compares it with the observed earthquake data. The user can specify whether to plot the cumulative
    counts or the event rates. The confidence intervals for the forecasts are also displayed.

    Args:
        dict_fore (dict): Dictionary containing forecast data organized by year, month, and forecast.
                           The forecast data should include the number of events.
        catalog (Catalog): The catalog of observed earthquakes. It must have a `to_dataframe()` method 
                           that returns a DataFrame with `origin_time` and a `get_magnitudes()` method for magnitudes.
        alpha_level (float, optional): The significance level for confidence intervals. Default is 0.05.
        type (str, optional): The type of plot. Can be 'cumulative' or 'rate'. Default is 'cumulative'.
        size_p (int, optional): The size of the plot points. Default is 10.
        forecast_label (str, optional): Label for the forecast in the plot legend. Default is 'forecast'.
        return_df (bool, optional): If True, return the concatenated DataFrame with forecast and observed data. Default is False.
        show_obs_counts (bool, optional): If True, display the observed earthquake counts on the rate plot. Default is False.

    Returns:
        None or pd.DataFrame:
            If `return_df` is True, the function will return a DataFrame containing both forecast and observed data.
            Otherwise, the function will only display the plot.

    Raises:
        ValueError: If the `type` argument is not 'cumulative' or 'rate'.
    """
    # Get the forecast data and Poisson confidence intervals
    N_fore = get_N_per_fore(dict_fore)
    N_ci = get_N_CI(N_fore, alpha_level)

    # Convert catalog timestamp to datetime objects
    timestamp_ms = catalog.to_dataframe()['origin_time']
    timestamp_s = timestamp_ms / 1000.0
    date_time = [datetime.fromtimestamp(t) for t in timestamp_s]

    # Calculate the cumulative number of observed events
    N_ci['obs_cumulative'] = [np.sum([d < N_fore['day'].iloc[idx] for d in date_time]) for idx in np.arange(0, len(N_fore['day']))]

    # Calculate the observed events per day
    N_per_day = [N_ci['obs_cumulative'].iloc[idx + 1] - N_ci['obs_cumulative'].iloc[idx] for idx in np.arange(0, len(N_fore['day']) - 1)]

    if type == 'cumulative':
        # Plot the cumulative forecast and observed events with confidence intervals
        plt.vlines(N_fore['day'], N_ci['N_cs_lower_bound'], N_ci['N_cs_upper_bound'], color='lightblue', zorder=1)
        plt.scatter(N_fore['day'], N_fore['cumsum'], label=forecast_label, zorder=2, s=size_p)
        plt.scatter(N_fore['day'], N_ci['obs_cumulative'], label='observed', zorder=3, s=size_p, color='black', marker='x')

        plt.xticks(fontsize=8, rotation=45)
        plt.ylabel('Cumulative number earthquakes')
        plt.legend()
        plt.show()

    elif type == 'rate':
        # Plot the rate of events and observed magnitudes
        coeff = catalog.get_magnitudes()**5  # Magnitude-based scaling for point sizes
        point_sizes = (coeff / np.max(coeff)) * size_p

        fig, ax1 = plt.subplots()

        # Plot the log of event rates for the forecast
        ax1.set_ylabel('log(Rate)', color='tab:blue')
        ax1.plot(N_fore['day'], np.log(N_fore['N']), label='forecast', zorder=2)

        # Optionally show observed earthquake counts
        if show_obs_counts:
            ax1.scatter(N_fore['day'][:-1].tolist(), N_per_day, s=point_sizes, m='x')

        ax1.tick_params(axis='y', labelcolor='tab:blue')

        # Plot the observed magnitudes on a second y-axis
        ax2 = ax1.twinx()
        ax2.set_ylabel('Magnitudes', color='tab:red')
        ax2.scatter(date_time, catalog.get_magnitudes(), color='tab:red', s=point_sizes, alpha=0.6)
        ax2.tick_params(axis='y', labelcolor='tab:red')

        for label in ax1.get_xticklabels():
            label.set_fontsize(8)
            label.set_rotation(45)

        fig.tight_layout()
        plt.title(forecast_label)
        plt.show()

    else:
        raise ValueError("Invalid type. Choose either 'cumulative' or 'rate'.")

    # Return the concatenated DataFrame if requested
    if return_df:
        return pd.concat([N_fore, N_ci], axis=1)

def get_N_plot_multi(dict_fore_array, catalog, forecast_label, alpha_level = 0.05, type = 'cumulative', size_p = 10,
                     return_df = False, save_plot = True, fig_dpi = 100, file_name = 'fig', 
                     show_obs_counts = False, show_uncertainty = False, ylims = None):
    """
    Plot the evolution of the number of events per day for multiple forecast models.

    This function generates a multi-forecast plot comparing the evolution of forecasted earthquake counts 
    over time with observed earthquake data. The user can specify whether to plot the cumulative counts 
    or the event rates, as well as how to display uncertainties and the observed earthquake magnitudes.

    Args:
        dict_fore_array (list of dicts): List of dictionaries, where each dictionary contains forecast data organized 
                                          by year, month, and forecast.
        catalog (Catalog): The catalog of observed earthquakes. It must have a `to_dataframe()` method that returns a 
                           DataFrame with `origin_time` and a `get_magnitudes()` method for magnitudes.
        forecast_label (list of str): List of labels corresponding to each forecast model to be plotted.
        alpha_level (float, optional): The significance level for confidence intervals. Default is 0.05.
        type (str, optional): The type of plot. Can be 'cumulative' or 'rate'. Default is 'cumulative'.
        size_p (int, optional): The size of the plot points. Default is 10.
        return_df (bool, optional): If True, return the concatenated DataFrame with forecast and observed data. Default is False.
        save_plot (bool, optional): If True, save the plot as a file. Default is True.
        fig_dpi (int, optional): The resolution (dots per inch) for the saved figure. Default is 100.
        file_name (str, optional): The name of the file to save the plot. Default is 'fig'.
        show_obs_counts (bool, optional): If True, display the observed earthquake counts on the rate plot. Default is False.
        show_uncertainty (bool, optional): If True, display the uncertainty bounds for the forecast rates. Default is False.
        ylims (tuple, optional): If provided, set the y-axis limits for the plot.

    Returns:
        None or pd.DataFrame:
            If `return_df` is True, the function will return a DataFrame containing both forecast and observed data.
            Otherwise, the function will only display the plot.

    Raises:
        ValueError: If the `type` argument is not 'cumulative' or 'rate'.
    """
    # Initialize color cycle for different forecasts
    colors = cycle(mcolors.TABLEAU_COLORS)

    # Create figure for rate plot (if not cumulative)
    if type != 'cumulative':
        fig, ax1 = plt.subplots()

    final_df = None

    # Loop through the list of forecast models
    for i, dict_fore in enumerate(dict_fore_array):
        # Get forecast data and Poisson confidence intervals
        N_fore = get_N_per_fore(dict_fore)
        N_ci = get_N_CI(N_fore, alpha_level)

        # Convert catalog timestamps to datetime
        timestamp_ms = catalog.to_dataframe()['origin_time']
        timestamp_s = timestamp_ms / 1000.0
        date_time = [datetime.fromtimestamp(t) for t in timestamp_s]

        # Calculate cumulative observed events
        N_ci['obs_cumulative'] = [np.sum([d < N_fore['day'].iloc[idx] for d in date_time]) for idx in np.arange(0, len(N_fore['day']))]

        # Calculate observed events per day
        N_per_day = [N_ci['obs_cumulative'].iloc[idx + 1] - N_ci['obs_cumulative'].iloc[idx] for idx in np.arange(0, len(N_fore['day']) - 1)]

        # Pick a color for this forecast model
        color = next(colors)

        # Plot for cumulative event counts
        if type == 'cumulative':
            plt.vlines(N_fore['day'], N_ci['N_cs_lower_bound'], N_ci['N_cs_upper_bound'], color=color, zorder=1, alpha=0.3)
            plt.scatter(N_fore['day'], N_fore['cumsum'], label=forecast_label[i], zorder=2, color=color, s=size_p)

        # Plot for event rates
        else:
            ax1.plot(N_fore['day'], np.log(N_fore['N']), label=forecast_label[i], zorder=2)

            # Plot uncertainty bounds if requested
            if show_uncertainty:
                fore_upper = [np.log(poisson_stat.ppf(0.975, N_fore_day)) for N_fore_day in N_fore['N']]
                ax1.plot(N_fore['day'], fore_upper, color=color, alpha=0.65, linestyle='--')
                ax1.set_ylabel('log(Daily number of earthquakes)')

            # Set the y-axis label for the rates plot
            else:
                ax1.set_ylabel('log(Daily number of earthquakes)', color='tab:blue')
                ax1.tick_params(axis='y', labelcolor='tab:blue')

        # Optionally return a DataFrame with forecast and observed data
        if return_df:
            temp = pd.concat([N_fore, N_ci], axis=1)
            temp['model'] = forecast_label[i]
            final_df = pd.concat([final_df, temp], axis=0)

    # Finalize the plot for cumulative data
    if type == 'cumulative':
        plt.scatter(N_fore['day'], N_ci['obs_cumulative'], label='$M$+ $3.95$ Observations', zorder=3, color='black', marker='x')
        plt.xticks(fontsize=8, rotation=45)
        plt.ylabel('Cumulative number of earthquakes')
        plt.legend(loc='lower right', frameon=True, fancybox=True, framealpha=0.2, facecolor='lightblue', edgecolor='grey', fontsize=12).get_frame().set_linewidth(2)

    # Finalize the plot for event rate data
    else:
        coeff = catalog.get_magnitudes() ** 5
        point_sizes = (coeff / np.max(coeff)) * size_p
        if show_obs_counts:
            ax1.scatter(N_fore['day'][:-1], np.log(np.array(N_per_day)), s=size_p, marker='x', zorder=3, color='black', label='$M \geq 3.95$ Observations')
        else:
            ax2 = ax1.twinx()
            ax2.set_ylabel('Magnitudes', color='tab:red')
            ax2.scatter(date_time, catalog.get_magnitudes(), color='tab:red', s=point_sizes, alpha=0.6)
            ax2.tick_params(axis='y', labelcolor='tab:red')

        for label in ax1.get_xticklabels():
            label.set_fontsize(8)
            label.set_rotation(45)

        fig.tight_layout()
        ax1.legend(frameon=True, fancybox=True, framealpha=0.2, facecolor='lightblue', edgecolor='grey', fontsize=12).get_frame().set_linewidth(2)

    # Return the concatenated DataFrame if requested
    if return_df:
        return final_df

    # Save the plot if requested
    if save_plot:
        plt.savefig(file_name, dpi=fig_dpi, bbox_inches='tight')

    # Show the plot
    plt.show()

def test_per_day(dict_fore, catalog, test_fun, plot_title = 'test', alpha_ = 0.05, plot_out = True,
                 save_plot = False, fig_dpi = 100, y_label = 'Daily test statistic', file_title = 'fig.png',  
                 ylims = None, size_p = 1, box_pos = (0.55, 0.5), n_cores = 10):
    """
    Run a statistical test for each day with at least one observation and plot the results.

    This function runs one of the CSEP tests (L, cond_L, M, or S) for each day where there is at least one 
    observation in the catalog. The test results are then plotted, and the user can choose to save the plot.
    The function also returns a DataFrame containing the test statistics for each day.

    Args:
        dict_fore (dict): A dictionary containing forecast data organized by year, month, and forecast.
        catalog (Catalog): The catalog of observed events. It must have a `filter()` method to filter events 
                           by time and an `to_dataframe()` method to convert to a DataFrame.
        test_fun (function): The statistical test function to run on the forecast and catalog data. It must return 
                             an object with `observed_statistic`, `test_distribution`, and `quantile` attributes.
        plot_title (str, optional): The title for the plot. Default is 'test'.
        alpha_ (float, optional): The significance level for the confidence interval. Default is 0.05.
        plot_out (bool, optional): If True, generate and show a plot of the test results. Default is True.
        save_plot (bool, optional): If True, save the plot to a file. Default is False.
        fig_dpi (int, optional): The resolution (dots per inch) for the saved figure. Default is 100.
        y_label (str, optional): The label for the y-axis. Default is 'Daily test statistic'.
        file_title (str, optional): The file name for saving the plot. Default is 'fig.png'.
        ylims (tuple, optional): If provided, set the y-axis limits for the plot.
        size_p (int, optional): The size of the points in the scatter plot. Default is 1.
        box_pos (tuple, optional): The position of the annotation box in the plot. Default is (0.55, 0.5).

    Returns:
        pd.DataFrame: A DataFrame containing the test statistics for each day. The columns include:
            - 'day': The date of the forecast.
            - 'observed_score': The observed test statistic for that day.
            - 'lower_q': The lower quantile of the test distribution.
            - 'upper_q': The upper quantile of the test distribution.
            - 'quantile': The quantile value for that day.
            - 'pass_condition': A boolean indicating whether the observed score is within the confidence interval.
    """
    # Initialize lists to store test results
    lowerq_per_day = []
    upperq_per_day = []
    obs_score_per_day = []
    date_fore_list = []
    quantile_per_day = []
    
    # Extract all forecast dates
    all_forecast_dates = extract_all_dates(dict_fore)
    min_date = min(all_forecast_dates)
    max_date = max(all_forecast_dates)

    def process_forecast(year, month, fore):
        tmp_fore = dict_fore[year][month][fore].copy()
        date_fore = datetime.strptime(extract_file_date(fore).replace('_', '-'), '%m-%d-%Y')
        start_date = str(date_fore.year) + '-' + str(date_fore.month) + '-' + str(date_fore.day) + ' 00:00:00.0'
        end_date = date_fore + timedelta(days=1)
        end_date = str(end_date.year) + '-' + str(end_date.month) + '-' + str(end_date.day) + ' 00:00:00.0'
        
        # Convert the start and end dates to epoch time
        start_epoch = csep.utils.time_utils.strptime_to_utc_epoch(start_date)
        end_epoch = csep.utils.time_utils.strptime_to_utc_epoch(end_date)
        
        # Filter the catalog for events that occurred during the forecast period
        filters = [f'origin_time >= {start_epoch}', f'origin_time < {end_epoch}']
        tmp_cat = catalog.filter(filters, in_place=False)

        tmp_filename = f'{year}_{month}_{extract_file_date(fore)}_tmp_fore_L.dat'
        
        # If there are observations for this forecast, run the test
        if tmp_cat.to_dataframe().shape[0] != 0:
            tmp_fore.to_csv(tmp_filename, sep='\t', header=None, index=None)
            fore_ = csep.load_gridded_forecast(tmp_filename, name='1')
            os.remove(tmp_filename)
            test_ = test_fun(fore_, tmp_cat)
            
            return {
                'date_fore': date_fore,
                'obs_score': test_.observed_statistic,
                'lower_q': np.quantile(test_.test_distribution, alpha_ / 2),
                'upper_q': np.quantile(test_.test_distribution, (1 - alpha_) / 2),
                'quantile': test_.quantile
            }
        return None

    results = Parallel(n_jobs=n_cores)(delayed(process_forecast)(year, month, fore) 
                                  for year in dict_fore.keys() 
                                  for month in dict_fore[year].keys() 
                                  for fore in dict_fore[year][month].keys())

    results = [res for res in results if res is not None]

    date_fore_list = [res['date_fore'] for res in results]
    obs_score_per_day = [res['obs_score'] for res in results]
    lowerq_per_day = [res['lower_q'] for res in results]
    upperq_per_day = [res['upper_q'] for res in results]
    quantile_per_day = [res['quantile'] for res in results]

    # Create a DataFrame with the test results
    df_output = pd.DataFrame({'day': date_fore_list,
                              'observed_score': obs_score_per_day,
                              'lower_q': lowerq_per_day,
                              'upper_q': upperq_per_day,
                              'quantile': quantile_per_day})

    # Determine if the test passed for each day
    df_output['pass_condition'] = np.logical_and(df_output['observed_score'] >= df_output['lower_q'], 
                                                 df_output['observed_score'] <= df_output['upper_q'])
    color_points = ['green' if condition else 'red' for condition in df_output['pass_condition']]
    
    # Calculate the percentage of missing days
    missing_perc = 1 - np.mean(df_output['pass_condition'])

    # Generate the plot if requested
    if plot_out:
        plt.scatter(df_output['day'], df_output['observed_score'], label='observed', zorder=3, s=size_p, c=color_points)
        plt.vlines(df_output['day'], df_output['lower_q'], df_output['upper_q'], color='black', zorder=1)
        plt.xticks(fontsize=8, rotation=45)
        plt.xlim(min_date, max_date)
        plt.annotate(plot_title + '\n' + fr'$\alpha = $ {alpha_:.2f}' + '\n' + fr'Red $\% = $  {missing_perc:.2f}', 
                     xy=(0.5, 0.5), xycoords='figure fraction', 
                     xytext=box_pos, textcoords='figure fraction', 
                     bbox=dict(boxstyle='round,pad=0.5', fc='lightblue', alpha=0.2),
                     fontsize=12, 
                     horizontalalignment='left', verticalalignment='bottom')
        plt.ylabel(y_label)
        if ylims is not None:
            plt.ylim(ylims[0], ylims[1])
        if save_plot:
            plt.savefig(file_title, dpi=300, bbox_inches='tight')
        plt.show()

    # Return the DataFrame containing the test results
    return df_output

def Ntest_per_day(dict_fore, catalog, alpha_ = 0.05, plot_out = True, plot_title = r'$\mathrm{N-test}$',
                  save_plot = False, file_title = 'fig.png', fig_dpi = 100, 
                  ylims = None, size_p = 1, box_pos = (0.55, 0.5), n_cores = 10):
    """
    Run the N-test (Poisson number test) for each day with at least one observation and plot the results.

    This function runs the N-test for each day where there is at least one observation in the catalog.
    The N-test compares the observed number of events to the expected number based on the Poisson distribution.
    The test results are then plotted, and the user can choose to save the plot. The function also returns 
    a DataFrame containing the test statistics for each day.

    Args:
        dict_fore (dict): A dictionary containing forecast data organized by year, month, and forecast.
        catalog (Catalog): The catalog of observed events. It must have a `filter()` method to filter events 
                           by time and an `to_dataframe()` method to convert to a DataFrame.
        alpha_ (float, optional): The significance level for the confidence interval. Default is 0.05.
        plot_out (bool, optional): If True, generate and show a plot of the test results. Default is True.
        plot_title (str, optional): The title for the plot. Default is r'$\mathrm{N-test}$'.
        save_plot (bool, optional): If True, save the plot to a file. Default is False.
        file_title (str, optional): The file name for saving the plot. Default is 'fig.png'.
        fig_dpi (int, optional): The resolution (dots per inch) for the saved figure. Default is 100.
        ylims (tuple, optional): If provided, set the y-axis limits for the plot.
        size_p (int, optional): The size of the points in the scatter plot. Default is 1.
        box_pos (tuple, optional): The position of the annotation box in the plot. Default is (0.55, 0.5).

    Returns:
        pd.DataFrame: A DataFrame containing the test statistics for each day. The columns include:
            - 'day': The date of the forecast.
            - 'observed_score': The observed number of events for that day.
            - 'lower_q': The lower quantile of the Poisson distribution.
            - 'upper_q': The upper quantile of the Poisson distribution.
            - 'quantile': The cumulative distribution function value for the observed score.
            - 'pass_condition': A boolean indicating whether the observed score is within the confidence interval.
    """
    # Initialize lists to store test results
    lowerq_per_day = []
    upperq_per_day = []
    obs_score_per_day = []
    quantile_per_day = []
    date_fore_list = []
    
    def process_forecast(year, month, fore):
        tmp_fore = dict_fore[year][month][fore].copy()
        date_fore = datetime.strptime(extract_file_date(fore).replace('_', '-'), '%m-%d-%Y')
        start_date = str(date_fore.year) + '-' + str(date_fore.month) + '-' + str(date_fore.day) + ' 00:00:00.0'
        end_date = date_fore + timedelta(days=1)
        end_date = str(end_date.year) + '-' + str(end_date.month) + '-' + str(end_date.day) + ' 00:00:00.0'
        
        start_epoch = csep.utils.time_utils.strptime_to_utc_epoch(start_date)
        end_epoch = csep.utils.time_utils.strptime_to_utc_epoch(end_date)
        
        filters = [f'origin_time >= {start_epoch}', f'origin_time < {end_epoch}']
        tmp_cat = catalog.filter(filters, in_place=False)

        tmp_filename = f'{year}_{month}_{extract_file_date(fore)}_tmp_fore_L.dat'
        
        if tmp_cat.to_dataframe().shape[0] != 0:
            tmp_fore.to_csv(tmp_filename, sep='\t', header=None, index=None)
            fore_ = csep.load_gridded_forecast(tmp_filename, name='1')
            os.remove(tmp_filename)
            test_ = poisson.number_test(fore_, tmp_cat)
            
            fore_rate = test_.test_distribution[1]
            return {
                'date_fore': date_fore,
                'obs_score': test_.observed_statistic,
                'lower_q': poisson_stat.ppf(alpha_/2, fore_rate),
                'upper_q': poisson_stat.ppf(1 - alpha_/2, fore_rate),
                'quantile': poisson_stat.cdf(test_.observed_statistic, fore_rate)
            }
        return None

    results = Parallel(n_jobs=n_cores)(delayed(process_forecast)(year, month, fore) 
                                  for year in dict_fore.keys() 
                                  for month in dict_fore[year].keys() 
                                  for fore in dict_fore[year][month].keys())

    results = [res for res in results if res is not None]

    date_fore_list = [res['date_fore'] for res in results]
    obs_score_per_day = [res['obs_score'] for res in results]
    lowerq_per_day = [res['lower_q'] for res in results]
    upperq_per_day = [res['upper_q'] for res in results]
    quantile_per_day = [res['quantile'] for res in results]

    # Create a DataFrame with the test results
    df_output = pd.DataFrame({'day': date_fore_list,
                              'observed_score': obs_score_per_day,
                              'lower_q': lowerq_per_day,
                              'upper_q': upperq_per_day,
                              'quantile': quantile_per_day})

    # Determine if the test passed for each day
    df_output['pass_condition'] = np.logical_and(df_output['observed_score'] >= df_output['lower_q'], 
                                                 df_output['observed_score'] <= df_output['upper_q'])
    color_points = ['green' if condition else 'red' for condition in df_output['pass_condition']]
    
    # Calculate the percentage of missing days
    missing_perc = 1 - np.mean(df_output['pass_condition'])

    # Generate the plot if requested
    if plot_out:
        plt.scatter(df_output['day'], df_output['observed_score'], label='observed', zorder=3, s=size_p, c=color_points)
        plt.vlines(df_output['day'], df_output['lower_q'], df_output['upper_q'], color='black', zorder=1)
        plt.xticks(fontsize=8, rotation=45)
        plt.annotate(plot_title + '\n' + fr'$\alpha = $ {alpha_:.2f}' + '\n' + fr'Red $\% = $  {missing_perc:.2f}', 
                     xy=(0.5, 0.5), xycoords='figure fraction', 
                     xytext=box_pos, textcoords='figure fraction', 
                     bbox=dict(boxstyle='round,pad=0.5', fc='lightblue', alpha=0.2),
                     fontsize=12, 
                     horizontalalignment='left', verticalalignment='bottom')
        if ylims is not None:
            plt.ylim(ylims[0], ylims[1])
        plt.ylabel('Daily number of earthquakes')
        if save_plot:
            plt.savefig(file_title, dpi=fig_dpi, bbox_inches='tight')
        plt.show()

    # Return the DataFrame containing the test results
    return df_output


def score_diff_per_day(dict_fore1, dict_fore2, catalog, 
                       score_fun = csep.utils.stats.get_Kagan_I1_score, n_cores = 10):
    """
    Calculate the difference in daily score differences between two models for each day with at least one observation.

    This function computes the Kagan Information score for two models for each day where there are observed events
    in the catalog. It then calculates the difference in the scores for the two models. The function returns a 
    DataFrame containing the dates and the corresponding score differences.

    Args:
        dict_fore1 (dict): A dictionary containing forecast data for the first model, organized by year, month, and forecast.
        dict_fore2 (dict): A dictionary containing forecast data for the second model, organized by year, month, and forecast.
        catalog (Catalog): The catalog of observed events. It must have a `filter()` method to filter events by time 
                           and an `to_dataframe()` method to convert to a DataFrame.
        score_fun (function, optional): A function to compute the score. The default is 
                                         `csep.utils.stats.get_Kagan_I1_score`.

    Returns:
        pd.DataFrame: A DataFrame containing the score difference for each day. The columns include:
            - 'score_diff': The difference in score between the two models.
            - 'day': The date of the forecast.
    """
    # Initialize lists to store the results
    date_fore_list = []
    score_diff = []
    
    def process_forecast(year, month, fore):
        date_str = extract_file_date(fore)
        date_fore = datetime.strptime(date_str.replace('_', '-'), '%m-%d-%Y')
        
        start_date = str(date_fore.year) + '-' + str(date_fore.month) + '-' + str(date_fore.day) + ' 00:00:00.0'
        end_date = date_fore + timedelta(days=1) 
        end_date = str(end_date.year) + '-' + str(end_date.month) + '-' + str(end_date.day) + ' 00:00:00.0'
        
        start_epoch = csep.utils.time_utils.strptime_to_utc_epoch(start_date)
        end_epoch = csep.utils.time_utils.strptime_to_utc_epoch(end_date)
        
        fore2_name = [k for k in dict_fore2[year][month].keys() if extract_file_date(k) == date_str][0]
        tmp_fore1 = dict_fore1[year][month][fore]
        tmp_fore2 = dict_fore2[year][month][fore2_name]
        
        filters = [f'origin_time >= {start_epoch}', f'origin_time < {end_epoch}']
        tmp_cat = catalog.filter(filters, in_place=False)

        tmp_filename1 = f'{year}_{month}_{extract_file_date(fore)}_tmp_fore1.dat'
        tmp_filename2 = f'{year}_{month}_{extract_file_date(fore2_name)}_tmp_fore2.dat'
        
        if tmp_cat.to_dataframe().shape[0] != 0:
            tmp_fore1.to_csv(tmp_filename1, sep='\t', header=None, index=None)
            tmp_fore2.to_csv(tmp_filename2, sep='\t', header=None, index=None)
            
            fore1 = csep.load_gridded_forecast(tmp_filename1, name='1')
            fore2 = csep.load_gridded_forecast(tmp_filename2, name='2')
            
            os.remove(tmp_filename1)
            os.remove(tmp_filename2)
            
            tmp_cat.filter_spatial(fore1.region)
            
            K_score1 = score_fun(fore1, tmp_cat)[0]
            K_score2 = score_fun(fore2, tmp_cat)[0]
            
            return {
                'date_fore': date_fore,
                'score_diff': K_score1 - K_score2
            }
        return None

    results = Parallel(n_jobs=n_cores)(delayed(process_forecast)(year, month, fore) 
                                  for year in dict_fore1.keys() 
                                  for month in dict_fore1[year].keys() 
                                  for fore in dict_fore1[year][month].keys())

    results = [res for res in results if res is not None]

    date_fore_list = [res['date_fore'] for res in results]
    score_diff = [res['score_diff'] for res in results]

    # Create a DataFrame with the results
    df_output = pd.DataFrame({'score_diff': score_diff,
                              'day': date_fore_list}) 
    
    # Return the DataFrame containing the score differences
    return df_output


def score_per_day(dict_fore1, catalog, score_fun = csep.utils.stats.get_Kagan_I1_score, n_cores = 10):
    """
    Calculate the difference in daily score differences between two models for each day with at least one observation.

    This function computes the Kagan Information score for two models for each day where there are observed events
    in the catalog. It then calculates the difference in the scores for the two models. The function returns a 
    DataFrame containing the dates and the corresponding score differences.

    Args:
        dict_fore1 (dict): A dictionary containing forecast data for the first model, organized by year, month, and forecast.
        dict_fore2 (dict): A dictionary containing forecast data for the second model, organized by year, month, and forecast.
        catalog (Catalog): The catalog of observed events. It must have a `filter()` method to filter events by time 
                           and an `to_dataframe()` method to convert to a DataFrame.
        score_fun (function, optional): A function to compute the score. The default is 
                                         `csep.utils.stats.get_Kagan_I1_score`.

    Returns:
        pd.DataFrame: A DataFrame containing the score difference for each day. The columns include:
            - 'score_diff': The difference in score between the two models.
            - 'day': The date of the forecast.
    """
    # Initialize lists to store the results
    date_fore_list = []
    score_diff = []
    
    def process_forecast(year, month, fore):
        date_str = extract_file_date(fore)
        date_fore = datetime.strptime(date_str.replace('_', '-'), '%m-%d-%Y')
        
        start_date = str(date_fore.year) + '-' + str(date_fore.month) + '-' + str(date_fore.day) + ' 00:00:00.0'
        end_date = date_fore + timedelta(days=1) 
        end_date = str(end_date.year) + '-' + str(end_date.month) + '-' + str(end_date.day) + ' 00:00:00.0'
        
        start_epoch = csep.utils.time_utils.strptime_to_utc_epoch(start_date)
        end_epoch = csep.utils.time_utils.strptime_to_utc_epoch(end_date)
        
        tmp_fore1 = dict_fore1[year][month][fore]
        
        filters = [f'origin_time >= {start_epoch}', f'origin_time < {end_epoch}']
        tmp_cat = catalog.filter(filters, in_place=False)

        tmp_filename1 = f'{year}_{month}_{extract_file_date(fore)}_tmp_fore1.dat'
        
        if tmp_cat.to_dataframe().shape[0] != 0:
            tmp_fore1.to_csv(tmp_filename1, sep='\t', header=None, index=None)
            
            fore1 = csep.load_gridded_forecast(tmp_filename1, name='1')
            
            os.remove(tmp_filename1)
            
            tmp_cat.filter_spatial(fore1.region)
            
            score1 = score_fun(fore1, tmp_cat)[0]
            
            return {
                'date_fore': date_fore,
                'score': score1 
            }
        return None

    results = Parallel(n_jobs=n_cores)(delayed(process_forecast)(year, month, fore) 
                                  for year in dict_fore1.keys() 
                                  for month in dict_fore1[year].keys() 
                                  for fore in dict_fore1[year][month].keys())

    results = [res for res in results if res is not None]

    date_fore_list = [res['date_fore'] for res in results if res is not None]
    score = [res['score'] for res in results if res is not None]

    # Create a DataFrame with the results
    df_output = pd.DataFrame({'score': score,
                              'day': date_fore_list}) 
    
    # Return the DataFrame containing the score differences
    return df_output

# fore_df is a gridbased forecat as pd.DataFrame (this are the files in the hdf5)
# cat_df is a pd.DataFrame of observations
# perc_retain is the percentage of observations used to calculate the score like average intensity where the top 5% of observations falls over overall average intensity
def calculate_S_score(fore_df, cat_df, perc_retain = 1):
    # this gets the spatial counts (essentially groups the space-magnitude bin according to the first 4 columns representing the spatial bins)
    spat_rates = fore_df.groupby([0,1,2,3], as_index=False)[8].sum()
    # get number of observations
    n_obs = cat_df.shape[0]
    n_selected = int(np.ceil(n_obs*perc_retain)) 
    
    # sort events by magnitude and select only the top perc_retain
    cat_selected = cat_df.sort_values(by='magnitude', ascending=False).head(n_selected)

    # initialise set of bins with observations
    bins_with_obs = []   
    # for each selected observation
    for obs_idx in np.arange(0, n_selected):
        # extract observation lon lat
        obs_long = cat_selected['longitude'].iloc[obs_idx]
        obs_lat = cat_selected['latitude'].iloc[obs_idx]
        # find the spatial bin
        flag_long = np.logical_and((spat_rates.iloc[:,0] <= obs_long), (spat_rates.iloc[:,1] >= obs_long) )
        flag_lat = np.logical_and((spat_rates.iloc[:,2] <= obs_lat), (spat_rates.iloc[:,3] >= obs_lat) )
        flag_bin = np.logical_and(flag_long, flag_lat)
        # get the rate
        spat_bin = spat_rates[flag_bin][:4]
        bins_with_obs.append(spat_bin.iloc[0])

    df_out = pd.DataFrame(bins_with_obs)
    df_out = df_out.drop_duplicates(subset=[0,1,2,3])
    return np.mean(df_out.iloc[:,4])/np.mean(fore_df.iloc[:,8])



def Kagan_and_S_score_per_day(dict_fore1, catalog, n_cores = 10):
    """
    Calculate the difference in daily score differences between two models for each day with at least one observation.

    This function computes the Kagan Information score for two models for each day where there are observed events
    in the catalog. It then calculates the difference in the scores for the two models. The function returns a 
    DataFrame containing the dates and the corresponding score differences.

    Args:
        dict_fore1 (dict): A dictionary containing forecast data for the first model, organized by year, month, and forecast.
        catalog (Catalog): The catalog of observed events. It must have a `filter()` method to filter events by time 
                           and an `to_dataframe()` method to convert to a DataFrame.
        

    Returns:
        pd.DataFrame: A DataFrame containing the score difference for each day. The columns include:
            - 'score_diff': The difference in score between the two models.
            - 'day': The date of the forecast.
    """
    # Initialize lists to store the results
    #date_fore_list = []
    
    def process_forecast(year, month, fore):
        date_str = extract_file_date(fore)
        date_fore = datetime.strptime(date_str.replace('_', '-'), '%m-%d-%Y')
        
        start_date = str(date_fore.year) + '-' + str(date_fore.month) + '-' + str(date_fore.day) + ' 00:00:00.0'
        end_date = date_fore + timedelta(days=1) 
        end_date = str(end_date.year) + '-' + str(end_date.month) + '-' + str(end_date.day) + ' 00:00:00.0'
        
        start_epoch = csep.utils.time_utils.strptime_to_utc_epoch(start_date)
        end_epoch = csep.utils.time_utils.strptime_to_utc_epoch(end_date)
        
        tmp_fore1 = dict_fore1[year][month][fore]
        
        filters = [f'origin_time >= {start_epoch}', f'origin_time < {end_epoch}']
        tmp_cat = catalog.filter(filters, in_place=False)

        tmp_filename1 = f'{year}_{month}_{extract_file_date(fore)}_tmp_fore1.dat'
        
        if tmp_cat.to_dataframe().shape[0] != 0:
            tmp_fore1.to_csv(tmp_filename1, sep='\t', header=None, index=None)
            
            fore1 = csep.load_gridded_forecast(tmp_filename1, name='1')
            
            os.remove(tmp_filename1)
            
            tmp_cat.filter_spatial(fore1.region)
            
            score1 = csep.utils.stats.get_Kagan_I1_score(fore1, tmp_cat)[0]
            score2 = calculate_S_score(fore_df = tmp_fore1,  cat_df = tmp_cat.to_dataframe(),  perc_retain = 1)
            score3 = get_Kagan_I1_score_nonlog(fore1, tmp_cat)[0]

            return {
                'date_fore': date_fore,
                'Kagan_score': score1,
                'S_score' : score2,
                'Kagan_score_nl': score3
            }
        return None

    results = Parallel(n_jobs=n_cores)(delayed(process_forecast)(year, month, fore) 
                                  for year in dict_fore1.keys() 
                                  for month in dict_fore1[year].keys() 
                                  for fore in dict_fore1[year][month].keys())

    results = [res for res in results if res is not None]

    date_fore_list = [res['date_fore'] for res in results if res is not None]
    K_score = [res['Kagan_score'] for res in results if res is not None]
    S_score = [res['S_score'] for res in results if res is not None]
    K_score_nl = [res['Kagan_score_nl'] for res in results if res is not None]


    # Create a DataFrame with the results
    df_output = pd.DataFrame({'day': date_fore_list,
                              'Kagan_score': K_score,
                              'S_score': S_score, 
                              'Kagain_score_nl': K_score_nl
                              }) 
    
    # Return the DataFrame containing the score differences
    return df_output




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


def get_Kagan_I1_score_nonlog(forecasts, catalog):
    """
    A program for scoring (I_1) earthquake-forecast grids by the methods of:
    Kagan, Yan Y. [2009] Testing long-term earthquake forecasts: likelihood methods
                         and error diagrams, Geophys. J. Int., v. 177, pages 532-542.

    Some advantages of these methods are that they:
        - are insensitive to the grid used to cover the Earth;
        - are insensitive to changes in the overall seismicity rate;
        - do not modify the locations or magnitudes of test earthquakes;
        - do not require simulation of virtual catalogs;
        - return relative quality measures, not just "pass" or "fail;" and
        - indicate relative specificity of forecasts as well as relative success.
    
    Written by Han Bao, UCLA, March 2021. Modified June 2021
    
    Note that: 
        (1) The testing catalog and forecast should have exactly the same time-window (duration)
        (2) Forecasts and catalogs have identical regions

    Args:
        forecasts:  csep.forecast or a list of csep.forecast (one catalog to test against different forecasts)
        catalog:    csep.catalog 
        
    Returns:
       I_1 (numpy.array): containing I1 for each forecast in inputs

    """
    ### Determine if input 'forecasts' is a list of csep.forecasts or a single csep.forecasts
    try:
        n_forecast = len(forecasts) # the input forecasts is a list of csep.forecast
    except:  
        n_forecast = 1             # the input forecasts is a single csep.forecast
        forecasts = [forecasts]

    # Sanity checks can go here
    for forecast in forecasts:
        if forecast.region != catalog.region:
            raise RuntimeError("Catalog and forecasts must have identical regions.")

    # Initialize array
    I_1 = np.zeros(n_forecast, dtype=np.float64)

    # Compute cell areas
    area_km2 = catalog.region.get_cell_area()
    total_area = np.sum(area_km2)
    
    for j, forecast in enumerate(forecasts):
        # Eq per cell per duration in forecast; note, if called on a CatalogForecast this could require computed expeted rates
        rate = forecast.spatial_counts()
        # Get Rate Density and uniform_forecast of the Forecast
        rate_den = rate / area_km2
        uniform_forecast = np.sum(rate) / total_area
        # Compute I_1 score
        n_event = catalog.event_count
        counts = catalog.spatial_counts()
        non_zero_idx = np.argwhere(rate_den > 0)
        non_zero_idx = non_zero_idx[:,0]
        I_1[j] = np.dot(counts[non_zero_idx], rate_den[non_zero_idx]) / n_event
    
    return I_1