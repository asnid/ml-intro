"""
LINEAR REGRESSION
Predict Google stock prices (for end of subsequent day)

@author: Adam
"""

import pandas as pd
import quandl, math, datetime
import numpy as np
from sklearn import preprocessing, cross_validation
from sklearn.linear_model import LinearRegression
import pickle # Serialization
import matplotlib.pyplot as plt # To plot stuff
from matplotlib import style    # To make plotted stuff look decent

style.use('ggplot')	# What decent plot style to use

## GET DATA

df = quandl.get('WIKI/GOOGL')   # df = "dataframe"
# print(df.head())    # Show beginning of data
df = df[['Adj. Open','Adj. High','Adj. Low','Adj. Close', 'Adj. Volume']]   # Keep these columns, remove others (some have less worth than others)

## DEFINE FEATURES
#  For linear regression, need to actively choose exact variables to regress on -- things that are likely to be important

df['HL_PCT'] = (df['Adj. High'] - df['Adj. Low']) / df['Adj. Low'] * 100         # High/low percent (sort of a measure of volatility)
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100  # Percent change in price over course of day

df = df[['Adj. Close','HL_PCT','PCT_change','Adj. Volume']]    # Only use these four features
df.fillna(-99999, inplace=True)	# Replace any unavailable data with this number (so it will be treated as an outlier) -- can't use NaN

## DEFINE LABEL: stock price at end of day

forecast_col = 'Adj. Close'
forecast_out = int(math.ceil(0.01 * len(df))) 	# Want to predict 1% (0.01) of the dataframe, currently (2017-11) 34 days in advance

df['label'] = df[forecast_col].shift(-forecast_out)		# Shift columns up: label ("correct" answer) is adjusted close price 1% (of total data time) into future
# df.dropna(inplace=True)		# Drop any rows for which there is no label

# Get features and labels ready

X = np.array(df.drop(['label'], 1))		# Features. Creates new dataframe with everything but label, converts it to numpy array, sets it to x
X = preprocessing.scale(X)	# Good to scale data... but then, need to rescale any new values along with all old values.
								# Can help with training and testing, but can add processing time. Often not done if speed really matters.

X_lately = X[-forecast_out:]	# The X-values that we don't have a y-value for (so can't be trained or tested on)								
X = X[:-forecast_out]			# The rest of the X-values (that we have a y-value for)

df.dropna(inplace = True)
y = np.array(df['label'])	# Label


## LEARN

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)	 	# Use 20% of data as test data

# Train classifier for first time
clf = LinearRegression(n_jobs = 10)	# Define classifier; n_jobs is for parallelism (-1 is max)
clf.fit(X_train, y_train)	# Fit line to training data

# Save trained classifier
with open('linreg.pickle', 'wb') as f:	# Open file with intention to write, call it 'f'
	pickle.dump(clf, f)						# Dump classifier

# Load trained classifier
# clf = pickle.load(open('linreg.pickle', 'rb'))	# Load trained classifier

accuracy = clf.score(X_test, y_test)	# See how well classifier works on test data (squared error)
print('Linear regression accuracy:', accuracy)


## PREDICT new y data based on X data

forecast_set = clf.predict(X_lately)	# Can pass single value or array of values (prediction per value)
print('Forecasted stock prices for next', forecast_out, 'days:')
print(forecast_set)


## PLOT VALUES

# Get dates

last_date = df.iloc[-1].name
last_unix = last_date.timestamp()

one_day = 86400	# Seconds in a day
next_unix = last_unix + one_day

# Populate dataframe with date values:
# 	For each forecast (and day)
#	Set forecast and day as values in dataframe (make future features NaN)

df['Forecast'] = np.nan 	# Set entire column to NaN for now
for i in forecast_set:
	next_date = datetime.datetime.fromtimestamp(next_unix)		
	next_unix += one_day
	df.loc[next_date] = [np.nan for _ in range(len(df.columns) - 1)] + [i]	# .loc references index of dataframe
																			# Rest is a list of values that are NaN for each field (since no data into future)
																			# + [i] adds one value from forecast set

df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()