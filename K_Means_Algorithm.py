# -*- coding: utf-8 -*-
"""
Created on Thu May 18 16:47:02 2017

@author: Renel Chesak
"""

'''This is a set of vectorized algorithms that use a document-term numpy array to create K-Means cluster centroids and cluster 
assignments.
These functions will allow you to cluster your documents, then use the model centroids in a classifer algorithm to classify 
unknown documents.'''

###############################################
# Import required libraries:
############################################### 
import numpy as np

###############################################
# Euclidean Distance Algorithm:
############################################### 
def distEucld(vecA, vecB):
    return sqrt(sum(power(vecA - vecB, 2))) #la.norm(vecA-vecB)


###############################################
# Cosine Similarity Algorithm:
############################################### 

#cosine similarity between 2 vectors:
def cosine_sim(vecA, vecB):
    #create normalized copies of each vector:
    vecA_norm = np.linalg.norm(vecA)
    vecB_norm = np.linalg.norm(vecB)
    #compute cosine similarity:
    sim = np.dot(vecA, vecB)/(vecA_norm * vecB_norm) #this should return a scalar float
    return sim

###############################################
# Vectorized Random Centroid Algorithm:
############################################### 

#create K random centers in your dataset to be used in your clustering algorithm:
def randCent(dataSet, k):
    # Where dataSet is a Numpy array, and k is the desired number of clusters.
    n = np.shape(dataSet)[1] #n will be the number of terms
    centroids = np.zeros((k,n), dtype=float) #creates a matrix of zeros with the shape:K rows by n columns
    for j in range(n): #iterate once for each term
        minJ = min(dataSet[:,j]) #minimum value in that term's column
        rangeJ = float(max(dataSet[:,j]) - minJ) #range of values in that term's column
        centroids[:,j] = minJ + rangeJ * np.random.rand(k) #set each row in that term's column to a different random number
        # that has been multiplied by the sum of min and range as a way of normalizing the random centers
    centroids = np.array(centroids) # ensure the matrix is a 2D Numpy array, as these are extremely fast for matrix manipulation
    return centroids 

###############################################
# Vectorized K Means Algorithm:
############################################### 

def K_Means(dataSet, k, max_iter=300, distMeas=cosine_sim, createCent=randCent):
    # Where dataSet is a Numpy array, k is the desired number of clusters, max_iter is the maximum number of loops (to avoid large or
    # infinite iterations), distMeas is the desired similarity measure (cosine similarity or Euclidean distance), and createCent is
    # the function which creates the initial randomized cluster centers.
    
    # Store the number of documents
    docs = np.shape(dataSet)[0]
    
    # Create an array of zeros to store 1) the cluster assignment and 2) the distance measure
    cluster = np.zeros((docs,2))
    
    # Call the randCent function to create a randomized set of k centroids
    Centroids = createCent(dataSet, k)
    
    # Initial values to start the while loop
    clusterChanged = True
    cur_iter = 0
    
    # While the cluster assignments are still changing and we have not reached the max iteration:
    while clusterChanged and cur_iter < max_iter:
        # Update the clusterChanged value so we don't run into an infinite loop.
        clusterChanged = False # note that there is code further down to change this to true if the clusters did change
        
        #assign the previous cluster to an object:
        previous_cluster = np.array(cluster, copy=True) #intially, this will be the random cluster starting points
        
        # Create an intital matrix of zeros to store similarity values:
        similarity = np.zeros((k,docs)) #this will eventually show you how similar each doc is to each cluster's centroid
        for j in range(k): #iterate once for each cluster (row)
            # Run cosine similarity as a numpy function along axis 1 (updates each doc's similarity value, one cluster at a time):
            similarity[j,:] = np.apply_along_axis(distMeas, 1, dataSet, Centroids[j,:])
            # Documentation: 
            # Execute func1d(a, *args) where func1d operates on 1-D arrays and 'a' is a 1-D slice of 'arr' along axis.       
            # Apply a function to 1-D slices along the given axis.
       
        # Capture both the cluster assigment and the similarity value. Note: max values => most similar. 
        cluster[:,0] = np.argmax(similarity, axis=0)         
        cluster[:,1] = np.amax(similarity, axis=0) 
        
        # If the clusters have not changed, break out of while loop
        if np.array_equal(cluster, previous_cluster):
            break 
        else:
            clusterChanged = True #sets this back to true if the cluster assinments are still changing
            # We only bother to recalculate centroid if there was a cluster reassignment
            for centroid in range(k): #iterate once for each cluster
                # Find the indicies of the documents that make up each cluster
                correct_cluster = np.where(cluster[:,0] == centroid)[0] 
                # Use the indicies to grab and store those documents separately
                docs_in_cluster = np.take(dataSet, correct_cluster, axis=0) 
                # If more than one document, take the mean as the new centroid (otherwise centroid is that one document)
                if len(docs_in_cluster) > 0: # TEST THIS CODE to see if the if statement can be removed
                    Centroids[centroid,:] = np.mean(docs_in_cluster, axis=0)
        cur_iter += 1
    return Centroids, cluster

###############################################
# Vectorized Classifier (based upon the K_Means Clusters):
###############################################   

def Nearest_Neighbor_Classifier(centroids, dataSet, max_iter=300, distMeas=cosine_sim):
    # Where centroids are those found from running K_Means(), dataSet is a Numpy array, max_iter is the maximum number of loops (to 
    # avoid large or infinite iterations), and distMeas is the desired similarity measure (cosine similarity or Euclidean distance).
    # Note: This function finds which cluster centroid each document is nearest to, and assign that document to that cluster. It is
    # essentially a K Nearest Neighbor algorithm with k=1.
    
    docs = dataSet.shape[0]
    k = centroids.shape[0]
    
    # Create an array of zeros to store 1) the cluster assignment and 2) the distance value
    cluster = np.zeros((docs,2))
    
    # Create an intital matrix of zeros to store similarity values:
    similarity = np.zeros((k,docs))
    
    for j in range(k): #iterate once for each cluster
        # Run your similarity function along axis 1 (updates each doc's similarity value, one cluster at a time):
        similarity[j,:] = np.apply_along_axis(func1d=distMeas, axis=1, dataSet, centroids[j,:])
    
    # Capture both the cluster assigment and the similarity value. Note: max values => most similar. 
    cluster[:,0] = np.argmax(similarity, axis=0)
    cluster[:,1] = np.amax(similarity, axis=0)
    
    # Transpose similarity to make it a document-by-cluster matrix:
    similarity_T = similarity.T
    
    # Create a slice of cluster that just contains the assignments:
    cluster_assignments = cluster[:, 0]
    
    # Make the cluster_assignments a 2D Numpy array:
    cluster_assignments_reshaped = cluster_assignments.reshape(-1, 1)

    # Create an array where the rows are the docs, the first column is the cluster assignment, and the next K columns are the 
    # similarities to each cluster:
    all_results = np.concatenate((cluster_assignments_reshaped, similarity_T), axis=1)
    
    return all_results, cluster_assignments
