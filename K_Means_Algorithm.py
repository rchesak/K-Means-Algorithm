# -*- coding: utf-8 -*-
"""
Created on Thu May 18 16:47:02 2017

@author: Renel
"""

###############################################
# Import required libraries:
############################################### 
import numpy as np

'''This is a set of vectorized algorithms that use a document-term numpy array to create K-Means cluster centroids and cluster 
assignments.
These functions will allow you to cluster your documents, then use the model centroids in a classifer algorithm to classify 
unknown documents.'''


###############################################
# Cosine Similarity Algorithm:
############################################### 

#cosine similarity between 2 vectors:
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
    n = np.shape(dataSet)[1] #n will be the number of terms
    centroids = np.zeros((k,n), dtype=float) #creates a matrix of zeros with the shape:K rows by n columns
    for j in range(n): #iterate once for each term
        minJ = min(dataSet[:,j]) #minimum value in that term's column
        rangeJ = float(max(dataSet[:,j]) - minJ) #range of values in that term's column
        centroids[:,j] = minJ + rangeJ * np.random.rand(k) #set each row in that term's column to a different random number
        # that has been multiplied by the sum of min and range (this must be some way of normalizing the center so that it will
        #be somewhat close to the actual data points)
    centroids = np.array(centroids)
    return centroids 

###############################################
# Vectorized K Means Algorithm:
############################################### 

def K_Means(dataSet, k, max_iter=300, distMeas=cosine_sim, createCent=randCent):
    # Store the number of documents
    docs = np.shape(dataSet)[0]
    
    # Create an array of zeros to store 1) the cluster assignment and 2) the distance measure
    cluster = np.zeros((docs,2))
    
    # Call the randCent function to create a randomized set of k centroids
    Centroids = createCent(dataSet, k)
    
    # Initial values to start the while loop
    clusterChanged = True
    cur_iter = 0
    
    # While the clusters are changing and we have not reached the max iteration:
    while clusterChanged and cur_iter < max_iter:
        # Update the clusterChanged value so we don't run into an infinite loop. DO YOU REALLY NEED THIS since there is a break?
        clusterChanged = False # note that there is code further down to change this to true if the clusters did change
        #assign the previous cluster to an object:
        previous_cluster = np.array(cluster, copy=True)
        # Create an intital matrix of zeros to store similarity values:
        similarity = np.zeros((k,docs)) #this will eventually show you how similar each doc is to each cluster's centroid
        for j in range(k): #iterate once for each cluster (row)
            # Run cosine similarity as a numpy function along axis 1 
            #(updates each doc's similarity value for one cluster at a time):
            similarity[j,:] = np.apply_along_axis(distMeas, 1, dataSet, Centroids[j,:])

#similarity[j,] = np.apply_along_axis(func1d=distMeas, axis=1, arr=dataSet, args=Centroids[j,:])
# numpy.apply_along_axis(func1d, axis, arr, *args, **kwargs)      
# Apply a function to 1-D slices along the given axis.
# Execute func1d(a, *args) where func1d operates on 1-D arrays and 'a' is a 1-D slice of 'arr' along axis.         
        
        # Max values = most similar. Capture both the cluster assigment and the similarity value
        cluster[:,0] = np.argmax(similarity, axis=0) #'np.argmax' returns the indices of the maximum similarity values for each
        #doc => the cluster it is most similar to
        #'cluster[:,0]' => for each row in the first column, value = row index of the most similar cluster (row index => cluster)
        
        cluster[:,1] = np.amax(similarity, axis=0) #'np.amax' returns the maximum value of an array or maximum along an axis.
        #'cluster[:,1]' => for each row in the second column, value = max sim value for that document
        
        # If the clusters have not changed, break out of while loop
        if np.array_equal(cluster, previous_cluster):
            break #breaks the while loop if the cluster assinments haven't changed
        else:
            clusterChanged = True #sets this back to true if the cluster assinments are still changing
            # We only bother to recalculate centroid if there was a reassignment
            for centroid in range(k):
                # Find the documents in each cluster
                correct_cluster = np.where(cluster[:,0] == centroid)[0]
                #'np.where' return elements, either from x or y, depending on condition.
                # Take the documents in each cluster
                docs_in_cluster = np.take(dataSet, correct_cluster, axis=0)
                # If more than one document, take the mean as the new centroid
                if len(docs_in_cluster) > 0:
                    Centroids[centroid,:] = np.mean(docs_in_cluster, axis=0)
        cur_iter += 1
    return Centroids, cluster

###############################################
# Vectorized Classifier based upon the K_Means Model:
 ###############################################   

def K_Means_Model_Classifier2(centroids, dataSet, max_iter=300, distMeas=cosine_sim):
    
    docs = dataSet.shape[0]
    k = centroids.shape[0]
    
    # Create an array of zeros to store 1) the cluster assignment and 2) the distance value
    cluster = np.zeros((docs,2))
    
    # Create an intital matrix of zeros to store similarity values:
    similarity = np.zeros((k,docs))
    
    for j in range(k): #iterate once for each cluster (row in 'similarity')
        # Run cosine similarity as a numpy function along axis 1 
        #(updates each doc's similarity value for one cluster at a time):
        similarity[j,:] = np.apply_along_axis(distMeas, 1, dataSet, centroids[j,:])
    
    # Max values = most similar. Capture both the cluster assigment and the similarity value
    cluster[:,0] = np.argmax(similarity, axis=0) #'np.argmax' returns the indices of the maximum similarity values for each
    #doc => the cluster it is most similar to
    #'cluster[:,0]' => for each row in the first column, value = row index of the most similar cluster (row index => cluster)

    cluster[:,1] = np.amax(similarity, axis=0) #'np.amax' returns the maximum value of an array or maximum along an axis.
    #'cluster[:,1]' => for each row in the second column, value = max sim value for that document
    
    #invert similarity to make it doc-cluster:
    similarity_T = similarity.T
    
    #create a slice of cluster that just contains the assignments:
    cluster_assignments = cluster[:, 0]
    
    #make the cluster_assignments 2D:
    cluster_assignments_reshaped = cluster_assignments.reshape(-1, 1)

    #create an array where the rows are the docs, the first column is the cluster assignment, and the next K columns are the 
    #similarities to each cluster:
    all_results = np.concatenate((cluster_assignments_reshaped, similarity_T), axis=1)
    
    return all_results, cluster_assignments
