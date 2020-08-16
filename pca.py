import csv
import numpy as np
import pandas as pd
from scipy import linalg as LA
import matplotlib.pyplot as plt


"""
load data
"""
def loadData():
    df = pd.read_csv('graph_data_continuous.csv', sep=',')
    return df.to_numpy()

"""
this function is to display first two columns of the dataset
"""
def display_first_two(array):
    np.random.seed(999)
    x1=array[:,0]
    x2=array[:,1]
    plt.scatter(x1,x2,c='red',alpha=0.5)
    plt.title('first_two.pdf')
    plt.show()

"""
this PCA is the main idea for question 4
I use numpy and scipy library only
"""
def PCA(data,num_components):
    # the center of the data has changed
    data -= np.mean(data, axis=0)

    # each row represents a sample and we claculate the covariance matrix
    Matrix=np.cov(data,rowvar=False)

    # calculate eigen values and eigen vectors
    values,vectors=LA.eig(Matrix)

    # get indices of first num_components largest eigen values
    key=np.argsort(values)[::-1][:num_components]

    # get actual eigen value and eigen vectors
    eigen_value=values[key]
    eigen_vector=vectors[:,key]

    # recalculate the matrix product and get final result
    res = np.dot(data, eigen_vector)
    return res,eigen_value,eigen_vector


"""
this function is different from PCA method 
I only need to change one line 
    key=np.argsort(values)[::-1][:num_components]
to 
    key = np.argsort(values)[:num_components]
in other words, only the sorting order needs to be changed here
"""
def bottom(data,num_components):
    data -= np.mean(data, axis=0)
    Matrix = np.cov(data, rowvar=False)
    values, vectors = LA.eigh(Matrix)
    key = np.argsort(values)[:num_components]
    eigen_value = values[key]
    eigen_vector = vectors[:, key]
    res = np.dot(data, eigen_vector)
    return res, eigen_value, eigen_vector


"""
this function is not written by numpy and scipy
it's from sklearn for test purpose
"""
def libraryPCA(data):
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    pca.fit_transform(data)
    return pca.transform(data)



filename='graph_data_continuous.csv'
array=loadData()

"""
question a
"""
display_first_two(array)


"""
question b
"""
matrix,eigen_value_b,eigen_vector_b=PCA(array,2)
x1=matrix[:,0]
x2=matrix[:,1]
plt.figure()
plt.scatter(x1, x2, c='red', alpha=0.5)
plt.title('top_two.pdf')
plt.show()

"""
question c
"""
data, eigen_value_c, eigen_vector_c =bottom(array, 2)
x3 = data[:, 0]
x4 = data[:, 1]
plt.figure()
plt.scatter(x3, x4, c='red', alpha=0.5)
plt.title('bottom_two.pdf')
plt.show()

# grid=libraryPCA(array)
# x5 = grid[:, 0]
# x6 = grid[:, 1]
# plt.figure()
# plt.scatter(x5, x6, c='red', alpha=0.5)
# plt.title('library.pdf')
# plt.show()





