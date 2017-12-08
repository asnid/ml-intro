# -*- coding: utf-8 -*-
"""
SUPPORT VECTOR MACHINE with KERNELS and SOFT MARGIN
Primarily using code from Mathieu Blondel, Sept. 2010
"""

import numpy as np
from numpy import linalg
import cvxopt
import cvxopt.solvers

## KERNELS

def linear_kernel(x, y):
    return np.dot(x, y)

def polynomial_kernel(x, y, p=3):
    return (1 + np.dot(x, y))**p

# Radial basis functions
def gaussian_kernel(x, y, sigma=5.0):
    return np.exp(-linalg.norm(x - y)**2 / (2 * (sigma**2)))


## SVM OBJECT
#  - C = strength of slack (None = hard margin)
class SupportVectorMachine:
    def __init__(self, kernel=linear_kernel, C=None):
        self.kernel = kernel
        if C is not None: self.C = float(C)
        else: self.C = C
        
    # Train method
    # - X = Input data (matrix of size samples-by-features)
    # - y = Classes of input data X
    def fit(self, X, y):
        n_samples, n_features = X.shape
        
        # Set up stuff for optimization
        # K = np.zeros((n_samples, n_samples))
        # for i in range(n_samples):
        #    for j in range(n_samples):
        #         K[i,j] = self.kernel(X[i], X[j]) 
        K = np.array([[self.kernel(X[i], X[j]) for j in range(n_samples)] for i in range(n_samples)])
        
        P = cvxopt.matrix(np.outer(y,y) * K)
        q = cvxopt.matrix(np.ones(n_samples) * -1)
        A = cvxopt.matrix(y, (1,n_samples))
        b = cvxopt.matrix(0.0)
        
        if self.C is None:  # Hard margin
            G = cvxopt.matrix(np.diag(np.ones(n_samples) * -1))
            h = cvxopt.matrix(np.zeros(n_samples))
        else:   # Soft margin
            G = cvxopt.matrix(np.vstack((np.diag(np.ones(n_samples) * -1), np.identity(n_samples))))
            h = cvxopt.matrix(np.hstack((np.zeros(n_samples), np.ones(n_samples) * self.C)))
        
        # Optimize
        solution = cvxopt.solvers.qp(P,q,G,h,A,b)
        a = np.ravel(solution['x'])     # Lagrange multipliers
        
        # Identify support vectors (with nonzero Lagrange multipliers)
        sv = a > 1e-5
        ind = np.arange(len(a))[sv]
        self.a = a[sv]
        self.sv = X[sv]
        self.sv_y = y[sv]
        print("%d support vectors out of %d points" % (len(self.a), n_samples))
        
        # Intercept
        self.b = 0
        for n in range(len(self.a)):
            self.b += self.sv_y[n]
            self.b -= np.sum(self.a * self.sv_y * K[ind[n],sv])
        self.b /= len(self.a)
        
        # Weight vector
        if self.kernel == linear_kernel:
            self.w  = np.zeros(n_features)
            for n in range(len(self.a)):
                self.w += self.a[n] * self.sv_y[n] * self.sv[n]
        else:
            self.w = None
        
    def project(self, X):
        if self.w is not None:
            return np.dot(X, self.w) + self.b
        else:
            y_predict = np.zeros(len(X))
            for i in range(len(X)):
                s = 0
                for a, sv_y, sv in zip(self.a, self.sv_y, self.sv):
                    s += a * sv_y * self.kernel(X[i], sv)
                y_predict[i] = s
            return y_predict + self.b
        
    # Test method
    # - X: input data to predict (matrix of size samples-by-features)
    def predict(self, X):
        return np.sign(self.project(X))
  

if __name__ == "__main__":
    import pylab as pl
    
    ## HELPER METHODS FOR TESTING DATA
    
    # Generate linearly separable data   
    def lin_sep_data():
        mean1 = np.array([0, 2])
        mean2 = np.array([2, 0])
        cov = np.array([[0.8, 0.6], [0.6, 0.8]])
         
        X1 = np.random.multivariate_normal(mean1, cov, 100)
        y1 = np.ones(len(X1))
        X2 = np.random.multivariate_normal(mean2, cov, 100)
        y2 = -np.ones(len(X2))
         
        return X1, y1, X2, y2
     
    # Generate not linearly separable data
    def not_lin_sep_data():
        mean1 = [-1, 2]
        mean2 = [1, -1]
        mean3 = [4, -4]
        mean4 = [-4, 4]
        cov = [[1.0, 0.8], [0.8, 1.0]]
        
        X1 = np.vstack((np.random.multivariate_normal(mean1, cov, 50), np.random.multivariate_normal(mean3, cov, 50)))
        y1 = np.ones(len(X1))
        X2 = np.vstack((np.random.multivariate_normal(mean2, cov, 50), np.random.multivariate_normal(mean4, cov, 50)))
        y2 = -np.ones(len(X2))
        
        return X1, y1, X2, y2
    
    # Generate data that is separable but has overlap (needs soft margin)
    def lin_overlap_data():
        mean1 = np.array([0, 2])
        mean2 = np.array([2, 0])
        cov = np.array([[1.5, 1.0], [1.0, 1.5]])
         
        X1 = np.random.multivariate_normal(mean1, cov, 100)
        y1 = np.ones(len(X1))
        X2 = np.random.multivariate_normal(mean2, cov, 100)
        y2 = -np.ones(len(X2))
         
        return X1, y1, X2, y2
    
    def split_train_test(X1, y1, X2, y2):
        X1_train = X1[:90]
        y1_train = y1[:90]
        X2_train = X2[:90]
        y2_train = y2[:90]
        
        X_train = np.vstack((X1_train, X2_train))
        y_train = np.hstack((y1_train, y2_train))
        
        X1_test = X1[90:]
        y1_test = y1[90:]
        X2_test = X2[90:]
        y2_test = y2[90:]
        
        X_test = np.vstack((X1_test, X2_test))
        y_test = np.hstack((y1_test, y2_test))
        
        return X_train, y_train, X_test, y_test
    
    # Plotting data with a linear boundary
    def plot_margin(X1_train, X2_train, clf):
        
        # Returns a point y such that [x,y] is on the line <w,x> + b = c
        def f(x, w, b, c=0):
            return (-w[0] * x - b + c) / w[1]
        
        pl.plot(X1_train[:,0], X1_train[:,1], 'ro')
        pl.plot(X2_train[:,0], X2_train[:,1], 'bo')
        pl.scatter(clf.sv[:,0], clf.sv[:,1], s=100, c='g')
        
        # Decision boundary
        a0 = -4; a1 = f(a0, clf.w, clf.b)
        b0 =  4; b1 = f(b0, clf.w, clf.b)
        pl.plot([a0, b0], [a1, b1], 'k')
        
        # Positive support vector
        a0 = -4; a1 = f(a0, clf.w, clf.b, 1)
        b0 =  4; b1 = f(b0, clf.w, clf.b, 1)
        pl.plot([a0, b0], [a1, b1], 'k--')
        
        # Negative support vector
        a0 = -4; a1 = f(a0, clf.w, clf.b, -1)
        b0 =  4; b1 = f(b0, clf.w, clf.b, -1)
        pl.plot([a0, b0], [a1, b1], 'k')
        
        pl.axis('tight')
        pl.show()
        
    # Plotting data with a nonlinear boundary (when unraveled from some kernel)
    def plot_contour(X1_train, X2_train, clf):
        pl.plot(X1_train[:,0], X1_train[:,1], 'ro')
        pl.plot(X2_train[:,0], X2_train[:,1], 'bo')
        pl.scatter(clf.sv[:,0], clf.sv[:,1], s=100, c='g')
        
        X1, X2 = np.meshgrid(np.linspace(-6,6,50), np.linspace(-6,6,50))
        X = np.array([[x1, x2] for x1,x2 in zip(np.ravel(X1), np.ravel(X2))])
        Z = clf.project(X).reshape(X1.shape)
        
        pl.contour(X1, X2, Z, [0.0], colors='k', linewidths=1, origin='lower')
        pl.contour(X1, X2, Z + 1, [0.0], colors='grey', linewidths=1, origin='lower')
        pl.contour(X1, X2, Z - 1, [0.0], colors='grey', linewidths=1, origin='lower')
        
        pl.axis('tight')
        pl.show()
    

    ## METHODS FOR TESTING DATA
    
    def test_lin():
        X1, y1, X2, y2 = lin_sep_data()
        X_train, y_train, X_test, y_test = split_train_test(X1, y1, X2, y2)
        
        clf = SupportVectorMachine()
        clf.fit(X_train, y_train)
        
        y_predict = clf.predict(X_test)
        correct_predictions = np.sum(y_predict == y_test)
        print('%d out of %d predictions are correct' % (correct_predictions, len(y_predict)))
        
        plot_margin(X_train[y_train==1], X_train[y_train==-1], clf)
        
    def test_not_lin():
        X1, y1, X2, y2 = not_lin_sep_data()
        X_train, y_train, X_test, y_test = split_train_test(X1, y1, X2, y2)
        
        clf = SupportVectorMachine(polynomial_kernel)
        clf.fit(X_train, y_train)
        
        y_predict = clf.predict(X_test)
        correct_predictions = np.sum(y_predict == y_test)
        print('%d out of %d predictions are correct' % (correct_predictions, len(y_predict)))
        
        plot_contour(X_train[y_train==1], X_train[y_train==-1], clf)
    
    def test_soft():
        X1, y1, X2, y2 = lin_overlap_data()
        X_train, y_train, X_test, y_test = split_train_test(X1, y1, X2, y2)
        
        clf = SupportVectorMachine(C=1000.1)
        clf.fit(X_train, y_train)
        
        y_predict = clf.predict(X_test)
        correct_predictions = np.sum(y_predict == y_test)
        print('%d out of %d predictions are correct' % (correct_predictions, len(y_predict)))
        
        plot_contour(X_train[y_train==1], X_train[y_train==-1], clf)
        
    ## DO STUFF
    
    # test_lin()
    # test_not_lin()
    test_soft()