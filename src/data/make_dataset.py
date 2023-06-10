# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point

df = pd.read_csv('../../data/raw/uber.csv')

df.info(verbose=True)


def process_df(df):
    """
    This function processes a DataFrame to:
    1. Convert 'pickup_datetime' column to datetime type.
    2. Create new columns 'pickup_point' and 'dropoff_point', converting longitude and latitude 
       to Point geometry.
    3. Transform points from CRS 4326 to CRS 3857.
    4. Compute distance in meters between pickup and dropoff points.
    5. Filter out only the required columns for the final DataFrame.

    :param df: Input DataFrame, should contain columns 'pickup_datetime', 'pickup_longitude',
               'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude', 'key', 'fare_amount',
               and 'passenger_count'.
    :return: Processed DataFrame with columns 'key', 'fare_amount', 'pickup_datetime',
             'passenger_count', 'pickup_point', 'dropoff_point', and 'distance_meter'.
    """
    # Convert datetime column to datetime type and time minus 5 houts
    df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])

    # Convert the lng and lat to point geometry
    df['pickup_point'] = df.apply(lambda x: Point(
        x['pickup_longitude'], x['pickup_latitude']), axis=1)
    df['dropoff_point'] = df.apply(lambda x: Point(
        x['dropoff_longitude'], x['dropoff_latitude']), axis=1)

    # Transform the point from crs 4326 to crs 3857
    df['pickup_point'] = gpd.GeoSeries(
        df['pickup_point'], crs=4326).to_crs(3857)
    df['dropoff_point'] = gpd.GeoSeries(
        df['dropoff_point'], crs=4326).to_crs(3857)

    # Calculate the distance between pickup and dropoff
    df['distance_meter'] = df.apply(
        lambda x: x['pickup_point'].distance(x['dropoff_point']), axis=1)

    # Keep only required columns in final dataframe
    df = df[['key', 'fare_amount', 'pickup_datetime',
             'passenger_count', 'pickup_point', 'dropoff_point',
             'distance_meter']]

    return df


df = process_df(df)

pd.to_pickle(df, "../../data/processed/processed.pkl")
