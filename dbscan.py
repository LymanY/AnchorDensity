# -*- coding: utf-8 -*-
"""
This is a simple implementation of DBSCAN intended to explain the algorithm.

@author: Chris McCormick
"""

import numpy as np
from rho import density_akd, density_fkd, density_naive, density_lc,density_rnn
from sklearn.neighbors import NearestNeighbors
from sklearn import preprocessing

def DBSCAN(data, eps, threshold, dm, nn_dict=None, anchor_dict=None):
    """
    Cluster the dataset `data` using the DBSCAN algorithm.
    
    MyDBSCAN takes a dataset `data` (a list of vectors), a threshold distance
    `eps`, and a required number of points `MinPts`.
    
    It will return a list of cluster labels. The label -1 means noise, and then
    the clusters are numbered starting from 1.
    """
 
    # This list will hold the final cluster assignment for each point in data.
    # There are two reserved values:
    #    -1 - Indicates a noise point
    #     0 - Means the point hasn't been considered yet.
    # Initially all labels are 0.    
    n = data.shape[0]
    d = data.shape[1]
    labels = [0]*n

    # C is the ID of the current cluster.    
    C = 0

    rho = None
    if dm == 'naive':
        rho = density_naive(data, nn_dict)
    elif dm == 'lc':
        rho = density_lc(data, nn_dict)
    elif dm == 'fkd':
        rho = density_fkd(data, nn_dict)
    elif dm == 'akd':
        rho = density_akd(data, anchor_dict)
    elif dm == 'rnn':
        rho = density_rnn(data, nn_dict)

    rho = preprocessing.robust_scale(rho)
    print(np.count_nonzero(rho > threshold))
    print(n)
    graph = NearestNeighbors(radius=eps).fit(data).radius_neighbors_graph(data).astype('int')
    
    # This outer loop is just responsible for picking new seed points--a point
    # from which to grow a new cluster.
    # Once a valid seed point is found, a new cluster is created, and the 
    # cluster growth is all handled by the 'expandCluster' routine.
    
    # For each point P in the Dataset data...
    # ('P' is the index of the datapoint, rather than the datapoint itself.)
    for P in range(0, n):
    
        # Only points that have not already been claimed can be picked as new 
        # seed points.    
        # If the point's label is not 0, continue to the next point.
        if not (labels[P] == 0):
           continue
        

        
        # If the number is below MinPts, this point is noise. 
        # This is the only condition under which a point is labeled 
        # NOISE--when it's not a valid seed point. A NOISE point may later 
        # be picked up by another cluster as a boundary point (this is the only
        # condition under which a cluster label can change--from NOISE to 
        # something else).
        if rho[P] < threshold:
            labels[P] = -1
        # Otherwise, if there are at least MinPts nearby, use this point as the 
        # seed for a new cluster.    
        else: 
           C += 1
           NeighborPts = regionQuery(data, P, graph)
           growCluster(data, labels, P, NeighborPts, C, eps, rho, threshold, graph)

    print(np.count_nonzero(np.array(labels) == -1))
    print(np.max(labels))
    # All data has been clustered!
    return labels, rho


def growCluster(data, labels, P, NeighborPts, C, eps, rho, threshold, graph):
    """
    Grow a new cluster with label `C` from the seed point `P`.
    
    This function searches through the dataset to find all points that belong
    to this new cluster. When this function returns, cluster `C` is complete.
    
    Parameters:
      `data`      - The dataset (a list of vectors)
      `labels` - List storing the cluster labels for all dataset points
      `P`      - Index of the seed point for this new cluster
      `NeighborPts` - All of the neighbors of `P`
      `C`      - The label for this new cluster.  
      `eps`    - Threshold distance
      `MinPts` - Minimum required number of neighbors
    """

    # Assign the cluster label to the seed point.
    labels[P] = C
    
    # Look at each neighbor of P (neighbors are referred to as Pn). 
    # NeighborPts will be used as a FIFO queue of points to search--that is, it
    # will grow as we discover new branch points for the cluster. The FIFO
    # behavior is accomplished by using a while-loop rather than a for-loop.
    # In NeighborPts, the points are represented by their index in the original
    # dataset.
    i = 0
    while i < len(NeighborPts):    
        
        # Get the next point from the queue.        
        Pn = NeighborPts[i]
       
        # If Pn was labelled NOISE during the seed search, then we
        # know it's not a branch point (it doesn't have enough neighbors), so
        # make it a leaf point of cluster C and move on.
        if labels[Pn] == -1:
           labels[Pn] = C
        
        # Otherwise, if Pn isn't already claimed, claim it as part of C.
        elif labels[Pn] == 0:
            # Add Pn to cluster C (Assign cluster label C).
            labels[Pn] = C
            
            # Find all the neighbors of Pn
            
            # If Pn has at least MinPts neighbors, it's a branch point!
            # Add all of its neighbors to the FIFO queue to be searched. 
            if rho[Pn] >= threshold:
                PnNeighborPts = regionQuery(data, Pn, graph)
                NeighborPts = NeighborPts + PnNeighborPts
            # If Pn *doesn't* have enough neighbors, then it's a leaf point.
            # Don't queue up it's neighbors as expansion points.
            #else:
                # Do nothing                
                #NeighborPts = NeighborPts               
        
        # Advance to the next point in the FIFO queue.
        i += 1        
    
    # We've finished growing cluster C!


def regionQuery(data, P, graph):
    """
    Find all points in dataset `data` within distance `eps` of point `P`.
    
    This function calculates the distance between a point P and every other 
    point in the dataset, and then returns only those points which are within a
    threshold distance `eps`.
    """
    n = data.shape[0]
    # neighbors = []
    # for i in range(n):
    #     if graph[P, i] == 1:
    #         neighbors.append(i)
    # neighbors = np.array(graph[P, :].todense()).flatten().nonzero()[0].tolist()
    neighbors = graph[P, :].toarray().flatten().nonzero()[0].tolist()
    return neighbors


    

