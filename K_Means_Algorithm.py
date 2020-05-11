# -*- coding: utf-8 -*-
"""
Created on Thu May 18 16:47:02 2017

@author: Renel Chesak
"""

# Import required libraries:
import numpy as np

class Train():
    '''
    With this class, you can cluster items with Train.KMeansClustering() and then pass the generated cluster centroids to 
    Train.NearestNeighborClassifier() to classify unseen data. The latter is essentially a K Nearest Neighbor algorithm with k=1.
    '''

    def distEucld(vecA, vecB):
        '''
        Parameters
            vecA : 1D Numpy array
            vecb : 1D Numpy array
        
        Returns 
            sim : float
                The euclidean distance between two vectors.        
        '''
        return sqrt(sum(power(vecA - vecB, 2))) 

    def cosineSim(vecA, vecB):
        '''
        Parameters
            vecA : 1D Numpy array
            vecb : 1D Numpy array        
        
        Returns 
            sim : float
                The cosine similarity between two vectors.
        '''
        #create normalized copies of each vector:
        vecA_norm = np.linalg.norm(vecA)
        vecB_norm = np.linalg.norm(vecB)
        #compute cosine similarity:
        sim = np.dot(vecA, vecB)/(vecA_norm * vecB_norm) #this should return a scalar float
        return sim

    def randCent(dataSet, k):
        '''
        Vectorized Random Centroid Algorithm. Selects `k` random centers in your dataset to be used in a clustering algorithm.
        
        Parameters
            dataSet : 2-D Numpy array
                The dataset you wish to cluster.
            k : int
                Desired number of random centers to select with the dataSet
        
        Returns
            centroids : 2-D Numpy array
                The centroid-dimension matrix.
        '''
        n = np.shape(dataSet)[1] #n will be the number of dimensions
        centroids = np.zeros((k,n), dtype=float) #creates a matrix of zeros with the shape:K rows by n columns
        for j in range(n): #iterate once for each dimension
            minJ = min(dataSet[:,j]) #minimum value in that dimension's column
            rangeJ = float(max(dataSet[:,j]) - minJ) #range of values in that dimension's column
            centroids[:,j] = minJ + rangeJ * np.random.rand(k) #set each row in that dimension's column to a different random number
            # that has been multiplied by the sum of min and range as a way of normalizing the random centers
        centroids = np.array(centroids) 
        return centroids 

    def KMeansClustering(dataSet, k, max_iter=300, distMeas=cosineSim, createCent=randCent):
        '''
        K Means Algorithm. Works with 2D Numpy arrays, as these are extremely fast for matrix manipulation.

        Parameters
            dataSet : 2-D Numpy array
                The dataset you wish to cluster. For document clustering, pass a doc-term matrix.
            k : int
                Desired number of random centers to select with the dataSet
            max_iter : int, default 300   
                The maximum number of loops (to avoid large or infinite iterations)
            distMeas : function, default cosine_sim
                Distance measure
            createCent : function, default randCent
                Random centroid generator
       
        Returns
            Centroids : 2-D Numpy array
                The centroid-dimension matrix. If clustering docs, this would be the centroid-term matrix.
            cluster : 2-D Numpy array
                Matrix of all individual's cluster assignment (column 1) and distance from its cluster's centroid (column 2)
        '''
        # Store the number of individuals (if document clustering, number of docs)
        n = np.shape(dataSet)[0]
        
        # Initialize an array of zeros to store 1) the cluster assignment and 2) the distance measure
        cluster = np.zeros((n, 2))
        
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
            similarity = np.zeros((k, n)) #this will eventually show you how similar each doc is to each cluster's centroid
            for j in range(k): #iterate once for each cluster (row)
                # Run cosine similarity as a numpy function along axis 1 (updates each doc's similarity value, one cluster at a time):
                similarity[j,:] = np.apply_along_axis(distMeas, 1, dataSet, Centroids[j,:])
        
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
                    # If more than one individual/document, take the mean as the new centroid (otherwise centroid is that one individual/document)
                    if len(docs_in_cluster) > 0: # TEST THIS CODE to see if the if statement can be removed
                        Centroids[centroid,:] = np.mean(docs_in_cluster, axis=0)
            cur_iter += 1
        return Centroids, cluster

    def printDocClusterSummary(centroids, cluster, top_n_terms, dt_arr, terms_arr):
        '''
        Prints summary of document clustering results.

        Parameters
            centroids : 2-D Numpy array
                The centroid-term matrix.
            cluster : 2-D Numpy array
                Matrix of all document's cluster assignment (column 1) and distance from its cluster's centroid (column 2)        
            top_n_terms : int
                The number of most frequent terms to print.
            dt_arr : 2-D Numpy array
                The doc-term dataset that you clustered.
            terms_arr : 2-D Numpy array
                An array containing the terms represented in `dt_arr`.                    
        '''
        k = centroids.shape[0]

        # take the cluster assignments from the KMeansClustering output:
        cluster_assignments = cluster[:, 0]
        #create a list of arrays containing the indicies of the documents that are in each cluster
        cluster_i_doc_idx = [np.where(cluster_assignments == i)[0] for i in range(k)]

        for i in range(k):
            #grab indices of docs in cluster i
            doc_idx = cluster_i_doc_idx[i]
            #grab cluster i's centroid
            centroid = centroids[i, :]

            print('---------------------------------------------------------------')
            print("Cluster: {}".format(i))

            print('\t-----------------------')
            print('\tNumber of docs in cluster: {}'.format(len(doc_idx)))
            print('\t-----------------------')

            #prepare the top n terms (the terms in that centroid which the highest TFxIDF)
            top_n_terms_idx = np.argpartition(centroid, -top_n_terms)[-top_n_terms:]       
            top_n_terms_idx_sorted = top_n_terms_idx[np.argsort(-centroid[top_n_terms_idx])]
            top_n_terms_sorted = terms_arr[top_n_terms_idx_sorted]

            #make the term indicies a numpy array:
            top_n_terms_idx_sorted_arr = np.array(top_n_terms_idx_sorted, dtype='int32')        
            #grab the docs (rows) in cluster i:
            cluster_i_DT = dt_arr[doc_idx, :]
            #grab the top N terms (columns) in cluster i:
            top_cluster_i_DT = cluster_i_DT[:, top_n_terms_idx_sorted_arr] 
            #get the raw docFreq for each term:
            docFreq_cluster_i = np.array([(top_cluster_i_DT!=0).sum(0)]) #counts nonzero entries for each term
            #convert that to the % of docs that contain that term (docs containing the term / docs in the cluster):
            docFreq_percent_cluster_i = (docFreq_cluster_i)/len(doc_idx)               

            print('\t-----------------------')
            print('\tTop N terms | % of docs in cluster that contain those terms')
            print('\t-----------------------')
            for i in range(top_n_terms):
                print('\t{:>11} | {:.1f}%'.format(top_n_terms_sorted[i], 100*(docFreq_percent_cluster_i[0][i])))

            print('---------------------------------------------------------------\n')

    def NearestNeighborClassifier(centroids, dataSet, max_iter=300, distMeas=cosineSim):
        '''
        Classifier. Works with 2D Numpy arrays, as these are extremely fast for matrix manipulation.
        This function finds which cluster centroid each individual is nearest to, and assigns that individual to that cluster. It is
        essentially a K Nearest Neighbor algorithm with k=1.   

        Parameters
            centroids : 2-D Numpy array
                The centroid-dimension matrix, generated from running `K_Means()`.        
            dataSet : 2-D Numpy array
                The dataset you wish to classify.
            max_iter : int, default 300   
                The maximum number of loops (to avoid large or infinite iterations)
            distMeas : function, default cosine_sim
                Distance measure
        
        Returns
            all_results : 2-D Numpy array
                Matrix of all individual's predicted cluster assignment (column 1), and the similarities to each cluster (columns 2:k+1, 
                where k is the number of clusters).            
            cluster_assignments : 2-D Numpy array
                Predicted cluster assignments.
        '''                
        n = dataSet.shape[0]
        k = centroids.shape[0]
        
        # Create an array of zeros to store 1) the cluster assignment and 2) the distance value
        cluster = np.zeros((n, 2))
        
        # Create an intital matrix of zeros to store similarity values:
        similarity = np.zeros((k, n))
        
        for j in range(k): #iterate once for each cluster
            # Run your similarity function along axis 1 (updates each doc's similarity value, one cluster at a time):
            similarity[j,:] = np.apply_along_axis(distMeas, 1, dataSet, centroids[j,:])
        
        # Capture both the cluster assigment and the similarity value. Note: max values => most similar. 
        cluster[:,0] = np.argmax(similarity, axis=0)
        cluster[:,1] = np.amax(similarity, axis=0)
        
        # Transpose similarity to make it a document-by-cluster matrix:
        similarity_T = similarity.T
        
        # Create a slice of cluster that just contains the assignments:
        cluster_assignments = cluster[:, 0]
        
        # Make the cluster_assignments a 2D Numpy array:
        cluster_assignments_reshaped = cluster_assignments.reshape(-1, 1)

        all_results = np.concatenate((cluster_assignments_reshaped, similarity_T), axis=1)
        
        return all_results, cluster_assignments
