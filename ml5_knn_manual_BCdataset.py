# -*- coding: utf-8 -*-
"""
K-NEAREST NEIGHBOURS ALGORITHM
Using UCI breast cancer data: http://archive.ics.uci.edu/ml/datasets/
	Class: 2 = benign, 4 = malignant
	Missing attributes denoted by '?'

@author: Adam
"""

import numpy as np
import warnings
from collections import Counter
import pandas as pd
import random

num_tests = 5

## NEAREST NEIGHBOURS ALGORITHM
def k_nearest(data, predict, k = 0): # Training data, point to find, number of nearest neighbours to find	
	n_points = 0
	for i in data:
		n_points += len(data[i])
		
	if k == 0:
		k = len(data) + 1
		warnings.warn('K set automatically to num_classes + 1')
	elif k <= len(data):
		warnings.warn('K is less than total number of classes')
	elif k >= n_points:
		warnings.warn('K is greater than total number of data points')
		k = n_points - 1
	
	# Get distance between 'predict' and each known point
	distances = []
	for group in data:
		for item in data[group]:
			distances.append([np.linalg.norm(np.array(predict) - np.array(item)), group])
	
	# Mark k smallest distances (and corresponding classes)		
	nearest_distances = []
	nearest_classes = []
	for i in sorted(distances)[:k]:	# "For i among the k smallest distances"
		nearest_distances.append(i[0])
		nearest_classes.append(i[1])
	
	# Take most-occurring class among the k
	most_common = Counter(nearest_classes).most_common(1)[0][0]
	confidence = Counter(nearest_classes).most_common(1)[0][1] / k
	
	return most_common, confidence


## GET DATA
	
df = pd.read_csv('breast_cancer_wisconsin.data')
df.replace('?',-99999, inplace = True)
df.drop(['id'], 1, inplace = True)

full_data = df.astype(float).values.tolist() 	# List of lists (also ensuring that everything is numerical, specifically floats)

test_size = 0.2


## RUN TESTS

accuracy = []

for test_number in range(num_tests):
	# Split up training and test sets
	random.shuffle(full_data)
	train_data = full_data[:-int(test_size * len(full_data))]
	test_data = full_data[-int(test_size * len(full_data)):]
	
	train_set = {2:[], 4:[]}
	for i in train_data:
		train_set[i[-1]].append(i[:-1])	# Append all features (not including class) to proper class in train_set
	test_set = {2:[], 4:[]}
	for i in test_data:
		test_set[i[-1]].append(i[:-1])		# Append all features (not including class) to proper class in test_set
	
	# Find nearest neighbours for each test data point
	correct_predictions = 0;
	unconfidence = [];
	for group in test_set:
		for test_point in test_set[group]:
			data_class, confidence = k_nearest(train_set, test_point)
			if data_class == group:
				correct_predictions += 1
			if confidence < 1:
				unconfidence.append(confidence)
	
	accuracy.append(correct_predictions / len(test_data))

np.set_printoptions(formatter={'float': lambda x: "{0:0.2f}".format(x)})
print('Average accuracy: %.3f' % (sum(accuracy) / len(accuracy)))
# print('There were %i less confident predictions:' % len(unconfidence), np.array(unconfidence))