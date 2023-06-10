
# perpare to clean the data and visualize it
import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
import matplotlib.pyplot as plt
import seaborn as sns

dat = pd.read_pickle("../../data/processed/processed.pkl")
dat.info(verbose=True)
dat.describe()
dat = (dat.replace([np.inf, -np.inf], np.nan)
       .dropna()
       .query("fare_amount > 0 and passenger_count > 0 and distance_meter > 0")
       .assign(distance_km=lambda df: (df['distance_meter'] / 1000).round(2)))


# shoe the three dimentional plot
sns.scatterplot(x="distance_km", y="fare_amount",
                hue="passenger_count", data=dat, palette="deep")
plt.show()

# drop the fare_amount = 500 and passenger_count = 208
# obviously, drive distance more than 100km and the price lower than 100 is not reasonable
dat = (dat.query("fare_amount < 400 and passenger_count < 208 and distance_km < 100"))

# create the columns log the distance_km
dat['log_distance_km'] = np.log(dat['distance_km'])
sns.scatterplot(x="log_distance_km", y="fare_amount", hue="passenger_count",
                data=dat[dat.distance_km < 10], palette="deep")
plt.show()

# drop the log_distance_km<-3 cause it seams not reasonable
dat = (dat.query("log_distance_km > -3"))

# create the columns log the fare_amount
dat['log_fare_amount'] = np.log(dat['fare_amount'])
sns.scatterplot(x="distance_km", y="log_fare_amount", hue="passenger_count",
                data=dat[dat.fare_amount < 10], palette="deep")

# drop the log_fare_amount<=1 cause it seams not reasonable
dat = (dat.query("log_fare_amount > 1"))

sns.scatterplot(x="distance_km", y="fare_amount", hue="passenger_count",
                data=dat[dat.fare_amount < 10], palette="deep")

# data cleaning is done


# each passenger_count has different fare_amount by box chart
sns.boxplot(x="passenger_count", y="fare_amount", data=dat[(
    dat['fare_amount'] < 30) & (dat['distance_km'] < 10)])
plt.show()
# seams like if the passenger_count is over 6, there are some bonus price for driver.

# set the two passenger count columns by passenger_count 1~5 and 6
dat['passenger_count_1_5'] = dat['passenger_count'].apply(
    lambda x: 1 if x in [1, 2, 3, 4, 5] else 0)
dat['passenger_count_6'] = dat['passenger_count'].apply(
    lambda x: 1 if x in [6] else 0)


# create the dow and hour columns
dat['dow'] = dat['pickup_datetime'].dt.dayofweek
dat['hour'] = dat['pickup_datetime'].dt.hour

# group by the dataset by dow to count the number of the observation and plot by histogram
dat.groupby('dow').size().plot(kind='bar')
dat.groupby('hour').size().plot(kind='bar')

# set the dummy variable for the dow, dow is 1, 2, 3 as a group which named weekday, 4,5 is Thu&Fri, 0, 6 is weekend
dat['weekday'] = dat['dow'].apply(lambda x: 1 if x in [1, 2, 3] else 0)
dat['Thu&Fri'] = dat['dow'].apply(lambda x: 1 if x in [4, 5] else 0)
dat['weekend'] = dat['dow'].apply(lambda x: 1 if x in [0, 6] else 0)


# set the dummy variable for the hour,
# hour between 1 to 6 is 'Pre-Sunrise Hours'
# hour between 7 to 16 is 'white_hours'
# hour between 17 to 23 is 'black_hours'

dat['pre_sunrise'] = dat['hour'].apply(
    lambda x: 1 if x in [1, 2, 3, 4, 5, 6] else 0)
dat['white_hours'] = dat['hour'].apply(
    lambda x: 1 if x in [7, 8, 9, 10, 11, 12, 13, 14, 15, 16] else 0)
dat['black_hours'] = dat['hour'].apply(
    lambda x: 1 if x in [17, 18, 19, 20, 21, 22, 23] else 0)


# show the year and month column and count it
# dataset is from 200901 to 201506
# never heard that uber have the seasonal pricing strategy, so didn't set the seasonal dummy columns


# create the categorical variable named time_category by 'Pre-Sunrise Hours', 'white_hours', 'black_hours'
# hour between 1 to 6 is 'Pre-Sunrise Hours'
# hour between 7 to 16 is 'white_hours'
# hour between 17 to 23 is 'black_hours'
dat['time_categories'] = dat['hour'].apply(lambda x: 'pre_sunrise' if x in [1, 2, 3, 4, 5, 6]
                                           else ('white_hours' if x in [7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
                                                 else 'black_hours'))


# plot the time of day to see the data trend
time_categories = ['pre_sunrise', 'white_hours', 'black_hours']

# Determine global min and max for 'distance_km' and 'fare_amount'
x_min, x_max = dat['distance_km'].min(), 40
y_min, y_max = dat['fare_amount'].min(), 100

fig, axs = plt.subplots(len(time_categories), 1,
                        figsize=(8, 6*len(time_categories)))

for i, category in enumerate(time_categories):
    data = dat[dat['time_categories'] == category]
    sns.scatterplot(x='distance_km', y='fare_amount', data=data, ax=axs[i])
    axs[i].set_title(f'{category}')
    axs[i].set_xlim([x_min, x_max])  # set the same x-axis limits
    axs[i].set_ylim([y_min, y_max])  # set the same y-axis limits

plt.tight_layout()
plt.show()


# create the categorical variable named dow_catagory by 'weekday','friday','saturday','sunday'
dat['dow_category'] = dat['dow'].apply(lambda x: 'weekday' if x in [1, 2, 3]
                                       else ('Thu&Fri' if x in [4, 5]
                                             else 'weekend'))


# plot the time of day to see the time trend
dow_category = ['weekday', 'Thu&Fri', 'weekend']

# Determine global min and max for 'distance_km' and 'fare_amount'
x_min, x_max = dat['distance_km'].min(), 40
y_min, y_max = dat['fare_amount'].min(), 100

fig, axs = plt.subplots(len(dow_category), 1, figsize=(8, 6*len(dow_category)))

for i, category in enumerate(dow_category):
    data = dat[dat['dow_category'] == category]
    sns.scatterplot(x='distance_km', y='fare_amount', data=data, ax=axs[i])
    axs[i].set_title(f'{category}')
    axs[i].set_xlim([x_min, x_max])  # set the same x-axis limits
    axs[i].set_ylim([y_min, y_max])  # set the same y-axis limits

plt.tight_layout()
plt.show()


dat.info(verbose=True)

test = dat[['fare_amount', 'distance_km', 'log_distance_km',
            'passenger_count_1_5', 'passenger_count_6',
            'weekday', 'Thu&Fri', 'weekend',
           'pre_sunrise', 'white_hours', 'black_hours'
            ]]
plt.figure(figsize=(12, 12))
dataplot = sns.heatmap(test.corr(), cmap="Blues", annot=True)
plt.show()

# select the 'log_distance_km' since the low correlation even it still a high correlation because the distance_km is the most important feature
# select the passenger_count_6 as the passenger_count and 0 means the baseline of lower than 5.

dat_model = dat[['fare_amount', 'log_distance_km', 'passenger_count_6',
                 'Thu&Fri', 'weekend', 'white_hours', 'black_hours']]

pd.to_pickle(dat_model, "../../data/external/external.pkl")
