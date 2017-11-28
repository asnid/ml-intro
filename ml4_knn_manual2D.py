# -*- coding: utf-8 -*-
"""
K-NEAREST NEIGHBOURS ALGORITHM

@author: Adam
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
from collections import Counter
import warnings

style.use('fivethirtyeight')

# Distance from p1 to p2 (both numpy arrays)
# Superceded by np.linalg.norm
def distance(p1, p2):
	return np.sqrt(np.sum((np.array(p2) - np.array(p1))**2))


## DATA

dataset = {'k':[[1,2],[2,3],[3,1]], 'r':[[6,5],[7,7],[8,6]]}	# Two classes and their features
new_features = [5,7]													# New feature (to figure out which class it belongs to)


[[plt.scatter(ii[0], ii[1], s = 100, color = i) for ii in dataset[i]] for i in dataset]
# Equivalent to:
# for i in dataset:
# 	for ii in dataset[i]:
# 		plt.scatter(ii[0], ii[1], s = 100, color = i)


## FIND CLASS ACCORDING TO K NEAREST NEIGHBOURS ALGORITHM
def k_nearest(data, predict, k = 0): # Training data, point to find, number of nearest neighbours to find
	n_points = 0
	for i in data:
		n_points += len(data[i])
		
	if k == 0:
		k = len(data) + 1
	elif k <= len(data):
		warnings.warn('K is less than total number of classes')
	elif k >= n_points:
		warnings.warn('K is greater than total number of data points')
	
	# Get distance between 'predict' and each known point
	distances = []
	for group in data:
		for item in data[group]:
			distances.append([distance(predict,item), group])
	
	# Mark k smallest distances (and corresponding classes)		
	nearest_distances = []
	nearest_classes = []
	for i in sorted(distances)[:k]:	# "For i among the k smallest distances"
		nearest_distances.append(i[0])
		nearest_classes.append(i[1])
	
	# Take most-occurring class among the k
	return Counter(nearest_classes).most_common(1)[0][0]


## TEST

result = k_nearest(dataset, new_features)
print('New features are in class(es):', result)

plt.scatter(new_features[0], new_features[1], s = 50, color = result)
plt.show()