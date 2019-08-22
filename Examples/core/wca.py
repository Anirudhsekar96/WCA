import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import scipy.stats as sc

def wca_mean(X, k, df):
	"""
		Implementes the WCA algorithm which maximizes the entropy with respect to the mean of the clusters

		X = Dataframe
		k = number of clusters
	"""
	

	# Intializing the clusters	
	C = dict()
	for cluster in range(k):
	    C[cluster] = pd.DataFrame()

	# Calculating the mean vector
	mean_vector = X.mean()

	# Choosing the seed points based on the minimum distance from the mean vector
	X['dist_mean'] = X.apply(lambda x: np.linalg.norm(np.asarray(x)- np.asarray(mean_vector)), axis=1)
	dist_means = X.sort_values(by='dist_mean')
	
	# Dropping the the datapoints which have already been assigned as seed
	idx_to_drop = dist_means.index[:k]
	dist_means.reset_index(drop=True,inplace=True)
	X.drop('dist_mean',axis=1,inplace=True)
	X.drop(idx_to_drop, inplace=True)

	# Assigning seed points to the clusters
	mu = list()
	for cluster in range(k):
	    C[cluster] = C[cluster].append(dist_means.iloc[cluster].drop('dist_mean'))
	    mu.append(C[cluster].mean())
	
	# Running the algorithm	
	
	# Initializing the p-value list which would be used for plotting
	pval = dict()

	for cluster in range(k):
	    pval[cluster] = dict()
	    for i in C[0].columns:
	        pval[cluster][i] = list()

	# Algorithm
	for i in tqdm(range(int(len(X)/k)), desc='Iterations: '):
	    for cluster in range(k):

	        # Calculating the distances from the mean vector of eaimportch cluster (in Descending order)
	        X['dist_mean'] = X.apply(lambda x: np.linalg.norm(np.asarray(x)- np.asarray(mu[cluster])), axis=1)
	        dist_means = X.sort_values(by='dist_mean', ascending=False)
	        idx_to_drop = dist_means.index[0]
	        dist_means.reset_index(drop=True,inplace=True)
	        X.drop('dist_mean',axis=1,inplace=True)

	        # Assigning the top value to the cluster
	        C[cluster] = C[cluster].append(dist_means.iloc[0].drop('dist_mean'))
	        C[cluster] = C[cluster].reset_index(drop=True)
	        
	        # Updating means of each cluster
	        mu[cluster] = C[cluster].mean()

	        # Remove datapoint from X?
	        X.drop(idx_to_drop,inplace=True)
	        
	        for i in C[0].columns:
	            pval[cluster][i].append(sc.ks_2samp(C[cluster][i],df.drop('target',axis=1)[i])[1])

	return(C,pval)
