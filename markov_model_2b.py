import numpy as np
import pandas as pd
import pdb
from functools import reduce
from scipy.stats import bernoulli, norm, laplace, expon, poisson, entropy
import matplotlib.pyplot as plt
from math import log
from scipy.stats import random_correlation

"""
Graph (a) one has the following edges
	g-e, r-e, e-s, e-t, t-l, l-o
Graph (b) as the following edges 
	g-e, r-e, t-e, s-t, s-o, l-o
"""

"""
this is the code professor provided
"""
def gradientDescent():
    df=pd.read_csv('graph_data_continuous.csv', sep=',')

    s = np.cov(df, rowvar=False)

    t = np.linalg.inv(s)
    t = s.copy()  # initializing t this way helps ensure stable behavior

    for i in range(1000):
        w = np.linalg.inv(t)
        t = t + .1 * (w - s - .2* np.sign(t))
    print(t)
    np.savetxt("lasso_out.csv", t, delimiter=",")
    np.savetxt("lasso_covar.csv", s, delimiter=",")


"""
this is the lagrange multiplier method I created for question 2
"""
def lagrangeMethod():
    df = pd.read_csv('graph_data_continuous.csv', sep=',')

    # each column represents a variable, with observations in the rows
    s = np.cov(df, rowvar=False)

    t = np.linalg.inv(s)
    t = s.copy()  # initializing t this way helps ensure stable behavior

    """
    we initialize a 7*7 lagrange matrix with random number between 0 and 1
    """
    lagrange=np.zeros((7, 7))

    """
    here I use a python list to store tuples which are edges in the graph
    """
    graph = [(0, 2), (1, 2), (2, 3), (3, 4), (4, 6), (5, 6)]

    for i in range(7):
        for j in range(7):
            if i == j or (i, j) in graph or (j, i) in graph:
                continue
            lagrange[i][j]=np.random.random()

    """
    number of epochs is 10000
    """
    for i in range(10000):
        w = np.linalg.inv(t)
        t = t + .1 * (w - s + .2 * lagrange)
        """
        because we have two gradient ascent formulas to deal with
        we should update parameters in lagrange matrix based on given graph structure
        
        Graph (b) as the following edges 
        g-e, r-e, t-e, s-t, s-o, l-o
        """
        for i in range(7):
            for j in range(7):
                if i==j or (i,j) in graph or (j,i) in graph:
                    continue
                lagrange[i][j]-=.1*(t[i][j])
    print(t)
    np.savetxt("lasso_out.csv", t, delimiter=",")
    np.savetxt("lasso_covar.csv", s, delimiter=",")

# gradientDescent()
lagrangeMethod()





