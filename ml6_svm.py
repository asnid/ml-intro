"""
SUPPORT VECTOR MACHINE
Using UCI breast cancer data: http://archive.ics.uci.edu/ml/datasets/
	Class: 2 = benign, 4 = malignant
	Missing attributes denoted by '?'
	
@author: Adam
"""

import numpy as np
from sklearn import preprocessing, model_selection, svm
import pandas as pd


## LOAD DATA

df = pd.read_csv('breast_cancer_wisconsin.data')

# Replace missing data with a very large-magnitude number -- algorithm will automatically treat it as an outlier instead of dumping data entirely
df.replace('?', -99999, inplace = True)
# Alternative (to actually drop data): df.dropna(inplace = True)

# Remove useless data (i.e., data that cannot help determine class)
df.drop(['id'], 1, inplace = True) 	# If this isn't dropped, classifier fails miserably (see by commenting this out)

X = np.array(df.drop(['class'], 1))  	# Features
y = np.array(df['class'])				  	# Label


## TRAIN CLASSIFIER

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size = 0.2)

clf = svm.SVC();	# SVC: "Support vector classifier"
clf.fit(X_train, y_train)


## TEST CLASSIFIER

accuracy = clf.score(X_test, y_test)
print('Classification accuracy:', accuracy)


## EXAMPLE PREDICTION

ex_measures = np.array([[4,2,1,1,1,2,3,2,1],[4,2,1,2,2,2,3,2,1]]) # Just make sure these 'samples' aren't in data document (so are new)
ex_measures = ex_measures.reshape(len(ex_measures), -1)	# To put numpy array into form that sklearn wants

prediction = clf.predict(ex_measures)
print('Prediction:', prediction)