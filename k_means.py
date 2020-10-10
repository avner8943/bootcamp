# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 21:47:29 2020

@author: avner
"""

from sklearn.datasets import load_iris
import numpy as np
from numpy import linalg
import matplotlib.pyplot as plt



# this function return vector of length points 
# each index repesenting point and the value is [0,k-1] is the point cluster
def clusters_update(points,means,k):
        
    dist = np.asarray([linalg.norm(points-means[i], axis=1) for i in range(k)])
    
    return np.argmin(dist,axis = 0)
    
    
    
# this function calculate the new means of the clusters    
def means_update(points, clusters,k):
    
    return np.asarray([np.mean(points[np.where(clusters==i)],axis = 0) for i in range(k)])
    

#this function draw the clusters    
def draw_clusters(points,clusters,means):
    
    x = points[:,0]
    y = points[:,1]
    plt.scatter(x, y, c = clusters)
    
    xx = means[:,0]
    yy = means[:,1]
    plt.scatter(xx, yy,c='r')
    
    plt.show()

#this is main while loop to check the convergence
def converge_means(points, means,k):
    
    thr = 1 # threshold of convergence for each coordinate
    
    while( np.any(thr >= 0.001)):
        
        clusters = clusters_update(points, means,k)
        
        new_means = means_update(points, clusters,k)
        
        thr = np.abs(np.divide(np.subtract(new_means , means),means)) # relative change
        
        means = new_means
        
        draw_clusters(points,clusters,means)
        
    return [means, clusters]
        

if __name__ == "__main__":

    iris_data = load_iris(return_X_y=True)

    points = iris_data[::2]
    
    points = points[0][:,:2]
    
    k = 3
    
    means_idx = np.random.choice(range(len(points)),k, replace=False)
    
    means = points[means_idx,:]     

    means, clusters = converge_means(points, means,k)
    



    
    
