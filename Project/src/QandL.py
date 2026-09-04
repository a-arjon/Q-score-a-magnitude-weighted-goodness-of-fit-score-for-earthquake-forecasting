import os
import pandas as pd
import numpy as np
import gc
import re
import math

def ell_score(fore_df, cat_df, waterlevel = 0):
    """
     Calculates the log-likelihood score of the expected rates of the forecast
    
     fore_df: gridbased forecat as pd.DataFrame 
     cat_df: pd.DataFrame of observations
     waterlevel: expected rate to add to zero valued bins in forecast.
    """

    #get the cumulative rate of the forecast, agriggated over the sapace-magnitude bins
    spat_mag_rates = fore_df.groupby([0,1,2,3,6,7], as_index=False)[8].sum()
    
    # get number of observations
    n_obs = cat_df.shape[0]

    # initialise set of bins with observations
    bins_with_obs = []
    # for each selected observation
    for obs_idx in np.arange(0, n_obs):
        # extract observation lon lat
        obs_long = cat_df['longitude'].iloc[obs_idx]
        obs_lat = cat_df['latitude'].iloc[obs_idx]
        # find the spatial bin
        flag_long = np.logical_and((spat_mag_rates.iloc[:,0] <= obs_long), (spat_mag_rates.iloc[:,1] >= obs_long) )
        flag_lat = np.logical_and((spat_mag_rates.iloc[:,2] <= obs_lat), (spat_mag_rates.iloc[:,3] >= obs_lat) )
        flag_bin = np.logical_and(flag_long, flag_lat)
        # get the rate
        spat_bin = spat_mag_rates[flag_bin][:6]
        bins_with_obs.append(spat_bin.iloc[0])

    #convert the forecast with the observed bins as a dataframe
    df_out = pd.DataFrame(bins_with_obs)
    # set zero bins to waterlevel
    target_idx = df_out.iloc[:,6] == 0
    df_out.iloc[target_idx,6] = waterlevel
    #take the natural log of the forecast rates on the observed bins
    log_rate = np.log(df_out.iloc[:,6]) 
    return log_rate.sum() -  spat_mag_rates.iloc[:, 6].sum()

###########################################################################################################

def combine_forecast_dataframes(nested_dict):
    """
    Converts the forecast directory into a single dataframe with the date of the forecast yyyy-mm-dd.

    Handles varied filename conventions like 'ETASV1_1_4_1_2014' and
    'ETAS_4_1_2014' safely using a year-indexing step-back approach.
    """
    all_dfs = []

    # Standard pyCSEP column mapping
    column_mapping = {
        0: "lon_min", "lon_min": "lon_min",
        1: "lon_max", "lon_max": "lon_max",
        2: "lat_min", "lat_min": "lat_min",
        3: "lat_max", "lat_max": "lat_max",
        6: "mag_min", "mag_min": "mag_min",
        7: "mag_max", "mag_max": "mag_max",
        8: "rate", "rate": "rate",
    }
    float_cols = ["lon_min", "lon_max", "lat_min", "lat_max", "mag_min", "mag_max", "rate"]
    target_cols = float_cols + ["date"]

    for year, months in nested_dict.items():
        for month, files in months.items():
            for file_name, df_forecast in files.items():
                # Avoid .copy(), modify the renamed frame directly
                df = df_forecast.rename(columns=column_mapping)

                # Extract all numbers from the filename as strings
                # e.g., ['1', '1', '4', '1', '2014'] or ['4', '1', '2014']
                numbers = re.findall(r"\d+", file_name)

                # Locate the year and step backward to grab month and day
                try:
                    year_idx = numbers.index(str(year))
                    month_val = numbers[year_idx - 2].zfill(2)
                    day_val = numbers[year_idx - 1].zfill(2)
                except (ValueError, IndexError):
                    # Fallback context mapping if regex fails or year is missing
                    month_val = str(month).zfill(2)
                    day_val = "01"

                # Construct exact matching timestamp date: 'YYYY-MM-DD'
                df["date"] = f"{year}-{month_val}-{day_val}"

                # Downcast numerical parameters to safeguard system RAM memory
                df[float_cols] = df[float_cols].astype("float32")

                # Keep only what we need
                all_dfs.append(df[target_cols])

    if not all_dfs:
        return pd.DataFrame(columns=target_cols)

    # Concat all frames at once
    master_df = pd.concat(all_dfs, ignore_index=True)

    # Convert to category after concatenation (massively faster)
    master_df["date"] = master_df["date"].astype("category")

    # Clean up memory completely at the very end
    del all_dfs
    gc.collect()

    return master_df



########################################################################################################
def catalog_to_direct(catalog):
    """
    converts the catalog to a directroy list, and prepares it for the q_score_calc() function
    """
    
    # Prepare day-precision catalog entries
    catalogEvents = catalog.to_dataframe()

    if 'origin_time' in catalogEvents.columns:
        # Explicit day-precision string matching (e.g., '2012-01-10')
        catalogEvents['date'] = pd.to_datetime(catalogEvents['origin_time'], unit='ms').dt.strftime('%Y-%m-%d')
    elif 'datetime' in catalogEvents.columns:
        catalogEvents['date'] = pd.to_datetime(catalogEvents['datetime']).dt.strftime('%Y-%m-%d')

    catalogEvents = catalogEvents.rename(columns={
        'mag': 'magnitude',
        'lon': 'longitude',
        'lat': 'latitude'
    })
    catalogEvents = catalogEvents.to_dict(orient='records')
    return catalogEvents
    
############################################################################################


def q_score_calc(forecast_df: pd.DataFrame, events: list[dict], alpha: float) -> float:
    if not events:
        raise ValueError("events must not be empty")
    if not 0 < alpha <= 1:
        raise ValueError("alpha must be in (0, 1]")
        
    selected_count = math.ceil(len(events) * alpha)
    selected_indices = sorted(
        range(len(events)),
        key=lambda index: float(events[index]["magnitude"]),
        reverse=True,
    )[:selected_count]
    
    top_events = [events[index] for index in selected_indices]
    rates_at_events = []
    
    eps = 1e-4
    
    for event in top_events:
        lon = float(event["longitude"])
        lat = float(event["latitude"])
        mag = float(event["magnitude"])
        
        mask = (
            (forecast_df["date"] == str(event["date"])) &
            (forecast_df["lon_min"] <= lon) & (lon <= forecast_df["lon_max"]) &
            (forecast_df["lat_min"] <= lat) & (lat <= forecast_df["lat_max"]) &
            (forecast_df["mag_min"] >= mag + eps) & (mag <= forecast_df["mag_max"] + eps)
        )
        
        matches = forecast_df.loc[mask, "rate"].values
        
        if len(matches) != 1:
            sub_df = forecast_df[forecast_df["date"] == str(event["date"])]
            spatial_mask = (
                (sub_df["lon_min"] - eps <= lon) & (lon <= sub_df["lon_max"] + eps) &
                (sub_df["lat_min"] - eps <= lat) & (lat <= sub_df["lat_max"] + eps) &
                (sub_df["mag_min"] - eps >= mag) & (mag <= sub_df["mag_max"] + eps)
            )
            fallback_matches = sub_df.loc[spatial_mask, "rate"].values
            if len(fallback_matches) >= 1:
                # Force extract the absolute first item as a scalar float
                val = float(fallback_matches[0])
            else:
                raise ValueError(f"event must match exactly one day/cell/magnitude-bin, got {len(matches)}: {event}")
        else:
            # Force extract the absolute first item as a scalar float
            val = float(matches[0])
                
        rates_at_events.append(val)
        
    print(selected_count)
    print(rates_at_events)
    print(selected_indices)
    
    rates_series = pd.to_numeric(forecast_df["rate"], errors='coerce')
    denominator = rates_series.mean()
    
    if denominator == 0 or pd.isna(denominator):
        raise ZeroDivisionError("mean ground rate for forecast is zero or invalid")
        
    print(denominator)
    numerator = sum(rates_at_events) / len(rates_at_events)
    
    return numerator / denominator


#######################################################################################

def calculate_Q_score(fore_df, cat_df, perc_retain = 1):
    """
     fore_df: gridbased forecat as pd.DataFrame
     cat_df: pd.DataFrame of observations
     perc_retain: the percentage of observations used to calculate the Q-score (Top alpha% of magnitude events)
    """
    
    # this gets the spatial counts (essentially groups the space-magnitude bin according to the first 4 columns representing the spatial bins and the 7 and 8 columns representing the magnitude bin)
    spat_rates = fore_df.groupby([0,1,2,3,6,7], as_index=False)[8].sum()
    # get number of observations
    n_obs = cat_df.shape[0]
    # get the number of events that are in above the 1 - alpha percentile
    n_selected = int(np.ceil(n_obs*perc_retain))
    # sort events by magnitude and select only the top perc_retain
    cat_selected = cat_df.sort_values(by='magnitude', ascending=False).head(n_selected)

    # initialise set of bins with observations from top perc_retain
    bins_with_obs = []
    # for each selected observation
    for obs_idx in np.arange(0, n_selected):
        # extract observation lon lat from top perc_retain events
        obs_long = cat_selected['longitude'].iloc[obs_idx]
        obs_lat = cat_selected['latitude'].iloc[obs_idx]
        # find the spatial bin
        flag_long = np.logical_and((spat_rates.iloc[:,0] <= obs_long), (spat_rates.iloc[:,1] >= obs_long) )
        flag_lat = np.logical_and((spat_rates.iloc[:,2] <= obs_lat), (spat_rates.iloc[:,3] >= obs_lat) )
        flag_bin = np.logical_and(flag_long, flag_lat)
        # get the rate
        spat_bin = spat_rates[flag_bin][:6]
        bins_with_obs.append(spat_bin.iloc[0])

    # dataframe with only top perc_retain forecast
    df_out = pd.DataFrame(bins_with_obs)
    forc_with_obs = []
    # extract all observations
    for obs_idx in np.arange(0, n_obs):
        # extract observation lon lat
        obs_long = cat_df['longitude'].iloc[obs_idx]
        obs_lat = cat_df['latitude'].iloc[obs_idx]
        # find the spatial bin
        flag_long = np.logical_and((spat_rates.iloc[:,0] <= obs_long), (spat_rates.iloc[:,1] >= obs_long) )
        flag_lat = np.logical_and((spat_rates.iloc[:,2] <= obs_lat), (spat_rates.iloc[:,3] >= obs_lat) )
        flag_bin = np.logical_and(flag_long, flag_lat)
        # get the rate
        spat_bin = spat_rates[flag_bin][:6]
        forc_with_obs.append(spat_bin.iloc[0])

    # dataframe with all observed forecast
    forc_out = pd.DataFrame(forc_with_obs)
    return np.mean(df_out.iloc[:,6])/np.mean(forc_out.iloc[:,6])

