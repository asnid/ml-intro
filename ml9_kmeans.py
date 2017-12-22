# -*- coding: utf-8 -*-
"""
K-MEANS ALGORITHM using sklearn
Titanic passenger dataset: https://www.youtube.com/redirect?redir_token=dezKm_7h6sxvzBxFuDq6bcRe0XV8MTUxMzEwNzYyMUAxNTEzMDIxMjIx&q=https%3A%2F%2Fpythonprogramming.net%2Fstatic%2Fdownloads%2Fmachine-learning-data%2Ftitanic.xls&event=video_description&v=8p6XaQSIFpY

Goal: try to cluster data to determine who survived the sinking

Features:
- pclass = Passenger Class (1/2/3 = 1st/2nd/3rd)
- survival = Survival (0/1 = no/yes)
- name = Name
- sex = Sex
- age = Age
- sibsp = Number of siblings/spouses aboard
- parch = Number of parents/children aboard
- ticket = Ticket number
- fare = Passenger fare (pounds)
- cabin = Cabin
- embarked = Port of embarkation (C/Q/S = Cherbourg/Queenstown/Southampton)
- boat = Lifeboat number
- body = Body ID number
- home.dest = Home/Destination
"""

import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import numpy as np
from sklearn import preprocessing, cross_validation
from sklearn.cluster import KMeans
import pandas as pd

## Random dataset to test clustering algorithm
# X = np.array([[1, 2], [1.5, 1.8], [5, 8], [8, 8], [1, 0.6], [9, 11]])
# plt.scatter(X[:,0], X[:,1], s=150, linewidths=5)
# plt.show()


## GET DATA

df = pd.read_excel('titanic.xls')

# Get rid of features that we don't care about
# - home.dest could help as a survival indicator, but needs more processing -- one-hot encoding will give way too many features
# - Classifier has better accuracy if omitting fare
df.drop(['body', 'name', 'ticket', 'home.dest', 'fare'], 1, inplace=True)

# Numericize and one-hot-encode text data that we care about
text_cols = ['sex', 'embarked', 'boat']
df_num = pd.get_dummies(df, columns = text_cols)

# Numericize cabin data based on deck layout
# Cabin number might matter also (e.g., cabins closer to stairs), but let's ignore that for now
cabins = [['T','A','B','C','D','E','F','G'], [0,1,2,3,4,5,6,7]]

cabin_num = []
for i in range(len(df['cabin'])):
    this_cabin = str(df['cabin'][i])
    try:
        cabin_index = cabins[0].index(this_cabin[0])
        cabin_num.append(cabins[1][cabin_index])
    except:
        cabin_num.append(-99999)        
df_num['cabin_num'] = cabin_num
df_num.drop('cabin', 1, inplace=True)

# Take care of unavailable data (put 0 in places)
df_num.fillna(0, inplace=True)


## K-MEANS CLASSIFIER

X = np.array(df_num.drop(['survived'], 1).astype(float))    # Features
X = preprocessing.scale(X)
y = np.array(df_num['survived'])


clf = KMeans(n_clusters=2)
clf.fit(X)

correct_predictions = 0
for i in range(len(X)):
    predict_me = np.array(X[i].astype(float))
    predict_me = predict_me.reshape(-1, len(predict_me))
    prediction = clf.predict(predict_me)
    
    if prediction[0] == y[i]:
        correct_predictions += 1

print("Percent Correct Predictions:", 100 * np.max([correct_predictions / len(X), 1 - correct_predictions / len(X)]))


## ====================== ALTERNATIVE:

# centroids = clf.cluster_centers_
# labels = clf.labels_

# Plot data and cluster centroids

# colors = 10 * ['g.','r.','c.','b.']    # Green Red Cyan Blue Black Orange

# for i in range(len(X)):
#     plt.plot(X[i][0], X[i][1], colors[labels[i]], markersize=15)

# plt.scatter(centroids[:,0], centroids[:,1], marker='x', s=150, linewidths=5, c='k')
# plt.show()