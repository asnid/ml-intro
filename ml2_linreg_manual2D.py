"""
2D LINEAR REGRESSION

@author: Adam
"""

from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import random

style.use('fivethirtyeight')


## RANDOM-ISH DATA

# xx = np.array([1,2,3,4,5,6], dtype = np.float64)
# yy = np.array([5,4,6,5,6,7], dtype = np.float64)

def create_dataset(num_points, variance, step = 1, correlation = False):
# Correlation: 'pos', 'neg', False (or anything else)
	start_val = 1
	xx = []
	yy = []
	for i in range(num_points):
		xx.append(i)
		yy.append(start_val + random.randrange(-variance, variance))
		
		if correlation == 'pos':
			start_val += step
		elif correlation == 'neg':
			start_val -= step
	
	return np.array(xx, dtype = np.float64), np.array(yy, dtype = np.float64)

xx, yy = create_dataset(40, 10, 2, 'pos')


## GET BEST-FIT EQUATION FOR LINE

def lin_fit(xx,yy):	
	m = ((mean(xx) * mean(yy)) - mean(xx * yy)) / (mean(xx)**2 - mean(xx * xx))
	b = mean(yy) - m * mean(xx)
	return m,b

m,b = lin_fit(xx,yy)
regression_line = [m * x + b for x in xx]
	# Same as:
	# for x in xx:
	#	regression_line.append(m*x + b)


# Plot data and line

plt.scatter(xx,yy)
plt.plot(xx, regression_line)
plt.show()


## ENSURE BEST-FIT LINE IS ALSO A GOOD FIT (R^2)

def squared_error(yy_orig, yy_line):
	return sum((yy_line - yy_orig)**2)

def coef_determination(yy_orig, yy_line):
	yy_mean_line = [mean(yy_orig) for y in yy_orig]
	return 1 - squared_error(yy_orig, yy_line) / squared_error(yy_orig, yy_mean_line)

r_squared = coef_determination(yy, regression_line)
print('Linear regression accuracy: R^2 = %.3f' % r_squared)


## MAKE PREDICTIONS

predict_x = 8
predict_y = m * predict_x + b
# print ('Predicted point: (%.2f,%.2f)' % (predict_x, predict_y))