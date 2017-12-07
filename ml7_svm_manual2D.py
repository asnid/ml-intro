# -*- coding: utf-8 -*-
"""
SUPPORT VECTOR MACHINE

@author: Adam
"""

import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np

style.use('ggplot')


## SVM OBJECT

class SupportVectorMachine:
    def __init__(self, visualization=True):
        self.visualization = visualization
        self.colours = {1:'r', -1:'b'} 	# Class 1 in red, class -1 in blue
        
        if self.visualization:
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(1,1,1) 	# 1x1 grid, 1st plot
	
    # Train method
    def fit(self, data):
        self.data = data 		# Save it (so entire algorithm can visualize it later)
        opt_dict = {}			# { |w|: [w,b] }
        
        transforms = [[1,1],[-1,1],[-1,-1],[1,-1]] 	# Will be applied to vector w in dot product
        		
        # Get max and min ranges to find start points for w and b
        all_data = []
        for yi in self.data:
            for featureset in self.data[yi]:
                for feature in featureset:
                    all_data.append(feature)
        self.max_feature_value = max(all_data)
        self.min_feature_value = min(all_data)
        all_data = None 	# Remove from memory
        
        # Set up optimization
        step_sizes = [self.max_feature_value * 0.1,
                      self.max_feature_value * 0.01,
                      self.max_feature_value * 0.001]
        b_range_multiple = 3    # b can take larger steps than w -- doesn't need to be as precise. (Make it lower to speed up the algorithm.)
        b_multiple =  5
        
        latest_optimum = self.max_feature_value * 10    # Starting value (first element in vector w)
        
        # Start stepping
        for step in step_sizes:
            w = np.array([latest_optimum, latest_optimum])  # Start at best point in last step
                                                            # Cutting a corner -- assuming w = [x,x] instead of w = [x,y]
            
            optimized = False
            while not optimized:
                for b in np.arange(-self.max_feature_value * b_range_multiple,  # Step through possible b values (want max. b -- most bias possible)
                                    self.max_feature_value * b_range_multiple,  # np.arange(start, end, step)
                                   step * b_multiple):                                 # Could use same step for b as we're doing with w, but it's more expensive and less necessary
                    
                    for t in transforms:
                        w_t = w * t;    # Which direction we're testing right now

                        found_option = True                     
                        for class_i in self.data:
                            for x_i in self.data[class_i]:
                                if not class_i * (np.dot(w_t, x_i) + b) >= 1:  # If constraint function not met for some sample,
                                    found_option = False                       # then this (w_t,b) is wrong -- don't keep it
                        
                        if found_option:
                            opt_dict[np.linalg.norm(w_t)] = [w_t,b]            # If all samples meet constraint functino, then (w_t,b) works -- keep it
                
                if w[0] < 0:            # If found best value at this step size,
                    optimized = True    # say so
                    print('Optimized a step.')
                else:                   # Otherwise,
                    w = w - step        # take a step and try again
            
            norms = sorted([n for n in opt_dict])   # Sorted list of all magnitudes
            opt_choice = opt_dict[norms[0]]         # Best choice is smallest norm (opt_dict entries look like |w|: [w,b])
            
            self.w = opt_choice[0]
            self.b = opt_choice[1]
            
            latest_optimum = opt_choice[0][0] + step * 2    # New start point for next level of stepping
            
            
        # How close to the support vector is each point?
        for class_i in self.data:
            for x_i in self.data[class_i]:
                print(x_i, ':', class_i * (np.dot(self.w, x_i) + self.b))
                # Could now say, "If one class doesn't have a point close enough to the support vector, then keep going" -- step again with a smaller step size
                    
    # Test method
    def predict(self, features):
        classification = np.sign(np.dot(np.array(features), self.w) + self.b)   # Predicted classification = sign(dot(x,w) + b)
        if classification != 0 and self.visualization:
            self.ax.scatter(features[0], features[1], s=200, marker='*', c=self.colours[classification])
        
        return classification 	# Is given data point above or below hyperplane?
    
    # Plot method
    def visualize(self):
        [[self.ax.scatter(x[0], x[1], s=100, c=self.colours[i]) for x in self.data[i]] for i in self.data]  # Plot all data points
        
        # Give values to plot on hyperplane (points where dot(x,w) + b = v = -1, 0, 1)
        def hyperplane(x,w,b,v):
            return (-w[0] * x - b + v) / w[1]
        
        data_range = (self.min_feature_value * 0.9, self.max_feature_value * 1.1)   # Size of graph
        hyp_x_min = data_range[0]
        hyp_x_max = data_range[1]
        
        psv1 = hyperplane(hyp_x_min, self.w, self.b, 1)     # Hyperplane for positive support vectors, point 1
        psv2 = hyperplane(hyp_x_max, self.w, self.b, 1)     # Hyperplane for positive support vectors, point 2
        self.ax.plot([hyp_x_min, hyp_x_max], [psv1, psv2], self.colours[1]) # colour: k = black
        
        nsv1 = hyperplane(hyp_x_min, self.w, self.b, -1)     # Hyperplane for negative support vectors, point 1
        nsv2 = hyperplane(hyp_x_max, self.w, self.b, -1)     # Hyperplane for negative support vectors, point 2
        self.ax.plot([hyp_x_min, hyp_x_max], [nsv1, nsv2], self.colours[-1])
        
        db1 = hyperplane(hyp_x_min, self.w, self.b, 0)     # Hyperplane for decision boundary, point 1
        db2 = hyperplane(hyp_x_max, self.w, self.b, 0)     # Hyperplane for decision boundary, point 2
        self.ax.plot([hyp_x_min, hyp_x_max], [db1, db2], 'y--') # colour: y-- = yellow
        
        plt.show()
        
        
        
## RANDOM TRAINING DATA

data_dict = {-1:np.array([[1,7],[2,8],[3,8],]),
              1:np.array([[5,1],[6,-1],[7,3]])}

svm = SupportVectorMachine()
svm.fit(data=data_dict)


# PREDICT

data_to_predict = [[0,10], [1,3], [3,4], [3,5], [5,5], [5,6], [6,-5], [5,8]]
for p in data_to_predict:
    svm.predict(p)
    
svm.visualize()