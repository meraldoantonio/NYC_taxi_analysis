# Helper functions
import pandas as pd
import numpy as np
from scipy.stats.mstats import winsorize
from scipy.stats import skew, skewtest

# viz
import matplotlib.pyplot as plt
import seaborn as sns
from haversine import haversine

# geoviz
from branca.element import Figure
import folium
from folium.plugins import FastMarkerCluster
import geohash as gh
from geopy.geocoders import Nominatim

# ML
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from pycaret.regression import * # import all as suggested by the documentation

# others
from tqdm import tqdm

# warnings
pd.options.mode.chained_assignment = None 
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)

def analyze_numeric_var(var, df, binwidth = 1, var_unit = None):   
    """
    Given a dataframe `df` containing a numeric variable `var`, perform the following:
        - Print out the summary statistics of `var`
        - Draw its distribution
        - Filter out rows in `df` whose values for `var` is less than a certain `min_val` or more than a certain `max_val`
    """
    var_series = df[var]
    var_med = np.median(var_series)
    var_mean = np.mean(var_series)
    
    winsorized_var_series = winsorize(df[var], limits=[0.01, 0.01])
    winsorized_var_med = np.median(winsorized_var_series)
    winsorized_var_mean = np.mean(winsorized_var_series)
    
    print(f"Summary statistics for `{var}`:")
    print(df[var].describe())
    
    (fig, ax) = plt.subplots(2,1, figsize=(15,10))
    ax[0] = sns.histplot(var_series,  ax=ax[0], kde=True, binwidth = binwidth)
    ax[0].axvline(var_med, lw=1.5, ls='dashed', color='black')
    ax[0].axvline(var_mean, lw=1.5, ls='dashed', color='red')
    ax[0].set_xlabel(f"`{var}` (mean: {var_mean:.2f}; median: {var_med:.2f})")
    
    ax[1] = sns.histplot(winsorized_var_series, ax=ax[1], kde=True, binwidth = binwidth)
    ax[1].axvline(winsorized_var_med, lw=1.5, ls='dashed', color='black')
    ax[1].axvline(winsorized_var_mean, lw=1.5, ls='dashed', color='red')
    ax[1].set_xlabel(f'`{var}` with the top/bottom 1% removed (mean: {winsorized_var_mean:.2f}; median: {winsorized_var_med:.2f})')
    if var_unit:
        fig.suptitle(f"Distribution of `{var}` in {var_unit}s")
    else:
        fig.suptitle(f"Distribution of `{var}`")
    plt.show()
    
    
def filter_numeric_var(var, df, min_val):
    """
    Given a dataframe `df` containing a numeric variable `var`, filter out rows in `df` whose values for `var` is less than or equal to a certain `min_val` 
    """  
    print(f"Minimum and maximum values of '{var}' for rows to be included (exclusive): {min_val}")
    print(f"Number of erroneous rows (`{var}` related): {len(df[df[var] <= min_val])}")
    df = df.loc[df[var] > min_val]
    print(f"Shape of the dataframe after filtering out erroneous rows (`{var}` related): {df.shape}\n")
    return df


geolocator = Nominatim(user_agent="ac")
def coord_to_addr(coord_tuple):
    """
    Given a tuple of latitude and longitude, convert it to human-readable address
    """
    coord_str = str(coord_tuple)[1:-1]
    addr = geolocator.reverse(coord_str).address
    short_addr = ",".join(addr.split(",")[:5])
    return short_addr



def manhattan(A, B):
    """  
    Given two coordinates A and B, calculate manhattan distance in km that takes into account the spherical shape of the earth
    """
    elbow_point = (A[0], B[1])
    dist_1 = haversine(A, elbow_point)
    dist_2 = haversine(elbow_point, B)
    manhattan_distance = dist_1 + dist_2
    return manhattan_distance


def analyze_trip_group(trip_group_id, df):
    """
    Given a trip-group-id, perform statistical analysis on it
    """
    assert "fare" in df.columns, "The `fare` column doesn't exist in the provided dataframe!"
    assert "duration" in df.columns, "The `duration` column doesn't exist in the provided dataframe!"
    
    
    _df = df.loc[df["trip_group_id"] == trip_group_id]
    fare = _df["fare"]
    duration = _df["duration"]
    
    fare_med = np.median(fare)
    fare_mean = np.mean(fare)
    
    duration_med = np.median(duration)
    duration_mean = np.mean(duration)
    
    fare_skewness_pval = skewtest(fare)[1]
    duration_skewness_pval = skewtest(duration)[1]
    
    fare_is_skewed = fare_skewness_pval > PVAL_THRESH
    duration_is_skewed = duration_skewness_pval > PVAL_THRESH
    
    
    fig, ax = plt.subplots(2,1, figsize=(15,10))
    ax[0] = sns.histplot(fare,  ax=ax[0], kde=True, binwidth = 1)
    ax[0].axvline(fare_med, lw=1.5, ls='dashed', color='black')
    ax[0].axvline(fare_mean, lw=1.5, ls='dashed', color='red')
    
    if fare_is_skewed:
        ax[0].set_xlabel(f"fare (skewness pval = {fare_skewness_pval:.3f}; is skewed)")
    else:
        ax[0].set_xlabel(f"fare (skewness pval = {fare_skewness_pval:.3f}; is not skewed)")
    
    ax[1] = sns.histplot(duration, ax=ax[1], kde=True, binwidth = 60)
    ax[1].axvline(duration_med, lw=1.5, ls='dashed', color='black')
    ax[1].axvline(duration_mean, lw=1.5, ls='dashed', color='red')
    if duration_is_skewed:
        ax[1].set_xlabel(f'duration (skewness pval = {duration_skewness_pval:.3f}; is skewed)')
    else:
        ax[1].set_xlabel(f"duration (skewness pval = {duration_skewness_pval:.3f}; is not skewed)")
    
    fig.suptitle(f"Distribution of `fare` and `duration` for trip group {str(trip_group_id)}\n median and mean are represented by black and red lines, respectively")
    
    
def mean_error(pred, true):
    """
    Given two Pandas series, `pred` and `true`, with equal length, return the mean of the difference between (pred - true) dataframe `df` containing a numeric variable `var`, perform the following:
    """
    return np.mean(pred - true)

def visualize_trips(geohash_start, geohash_end, df, line_color = "#ADD8E6"):
    """
    Given two geohashes and the df containing trips to and from these geohashes, visualize these trips.
    """
    # main map
    fig=Figure(width=1000,height=600)
    gh_limits_start = gh.bbox(geohash_start) 
    gh_limits_end = gh.bbox(geohash_end) 
    
    DELTA = 0.0002 
    bounds_start = [(gh_limits_start["n"], gh_limits_start["w"]+DELTA),# so that the boxes don't overlap
              (gh_limits_start["s"], gh_limits_start["w"]+DELTA),
              (gh_limits_start["n"], gh_limits_start["e"]),
              (gh_limits_start["s"], gh_limits_start["e"])]

    
    bounds_end = [(gh_limits_end["n"], gh_limits_end["w"]),
              (gh_limits_end["s"], gh_limits_end["w"]),
              (gh_limits_end["n"], gh_limits_end["e"]),
              (gh_limits_end["s"], gh_limits_end["e"])]
    
    
    centerpoint = ((gh_limits_start["n"] + gh_limits_end["n"])/2 + (gh_limits_start["s"] + gh_limits_end["s"])/2)/2,((gh_limits_start["w"] + gh_limits_end["w"])/2 + (gh_limits_start["e"] + gh_limits_end["e"])/2)/2
                   
    
    
    m = folium.Map(
        location=centerpoint,
        zoom_start=14.5,
        zoom_control=False,
        scrollWheelZoom=False,
        dragging=True

    )
        
    folium.Rectangle(bounds=bounds_start, color="#00ff00", fill=True, fill_color="#f2e1f1", fill_opacity=0.5).add_to(m)
    folium.Rectangle(bounds=bounds_end, color="#FF7F7F", fill=True, fill_color="#f2e1f1", fill_opacity=0.5).add_to(m)

    for idx, row in df.loc[df["trip_group_id"] == (geohash_start, geohash_end)].iterrows():
        points = [(row["start_lat"], row["start_long"]), (row["end_lat"], row["end_long"])]
        folium.PolyLine(points, 
                        weight=5,
                        color=line_color,
                        opacity=0.3).add_to(m)


    fig.add_child(m)
    return fig