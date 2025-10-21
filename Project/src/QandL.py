import os
import pandas as pd
import numpy as np


def ell_score(fore_df, cat_df, waterlevel = 0):
    # Calculates the log-likelihood score of the expected rates of the forecast
    
    ## fore_df: gridbased forecat as pd.DataFrame 
    ## cat_df: pd.DataFrame of observations
    ## waterlevel: expected rate to add to zero valued bins in forecast.


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




def calculate_Q_score(fore_df, cat_df, perc_retain = 1):
    # fore_df: gridbased forecat as pd.DataFrame
    # cat_df: pd.DataFrame of observations
    # perc_retain: the percentage of observations used to calculate the Q-score (Top alpha% of magnitude events)
    
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

