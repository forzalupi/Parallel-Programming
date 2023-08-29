#!/usr/bin/env python3
#
# File: kmeans.py
# Author: Originally by Alexander Schliep (alexander@schlieplab.org), updated by Matti Karppa (karppa@chalmers.se)
# 
# Requires scikit-learn (sklearn), numpy, matplotlib
#

import logging
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
import time
import multiprocessing as mp
from amdahl import amdahls_plot

def generateData(n, c):
    """generates Gaussian blobs, 
    see https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_blobs.html
    """
    logging.info(f"Generating {n} samples in {c} classes")
    X, y = make_blobs(n_samples=n, centers = c, cluster_std=1.7, shuffle=False,
                      random_state = 2122)
    return X


def nearestCentroid(datum, centroids):
    """computes the distance of the data vector to the centroids and returns 
    the closest one as an (index,distance) pair
    """
    # norm(a-b) is Euclidean distance, matrix - vector computes difference
    # for all rows of matrix
    dist = np.linalg.norm(centroids - datum, axis=1)
    return np.argmin(dist), np.min(dist)


def worker(data_chunk, centroids, k):
    """
    Assign clusters for each data point in the data chunk.
    """
    N = len(data_chunk)

    c = np.zeros(N, dtype=int) 
    variation = np.zeros(k)

    for i in range(N):
        cluster, dist = nearestCentroid(data_chunk[i], centroids)
        c[i] = cluster
        variation[cluster] += dist**2
    
    return c, variation


def recompute_centroids(cluster):
    """
    Recomputes the centroid of a given cluster.
    """
    # Extract x and y values
    x_values = cluster[:,0]
    y_values = cluster[:,1]

    # Take the mean of both axis
    mean_x = np.mean(x_values)
    mean_y = np.mean(y_values)

    # New centroid
    centroid = np.array([[mean_x, mean_y]])

    return centroid


def kmeans(k, data, nr_iter = 100, nr_workers=1):
    """computes k-means clustering by fitting k clusters into data
    a fixed number of iterations (nr_iter) is used
    you should modify this routine for making use of multiple threads
    """
    N = len(data)
    
    np.random.seed(1337)
    # Choose k random data points as centroids
    centroids = data[np.random.choice(np.array(range(N)),size=k,replace=False)]
    data_chunks = np.array_split(data, nr_workers)

    logging.debug("Initial centroids\n", centroids)
    logging.info("Iteration\tVariation\tDelta Variation")

    for j in range(nr_iter):
        logging.debug("=== Iteration %d ===" % (j+1))

        # Initialize
        cluster_sizes = np.zeros(k, dtype=int)
        c = np.empty((0,), dtype=int)
        total_variation = 0.0

        # Start workers
        with mp.Pool(nr_workers) as p:
            args_list = [(data_chunks[i], centroids, k) for i in range(nr_workers)]
            results = p.starmap(worker, args_list)
            # Loop over results list to unpack values
            for t in results:
               # The cluster index: c[i] = j indicates that i-th datum is in j-th cluster
               # Concatenate all arrays into one
                c = np.concatenate((c, t[0]), dtype=int)
                
                delta_variation = -total_variation
                # Sum the cluster variation from all the workers
                total_variation += np.sum(t[1])
                delta_variation += total_variation

        logging.info("%3d\t\t%f\t%f" % (j, total_variation, delta_variation))

        # Recompute centroids (parallelized)
        centroids = np.zeros((k,2)) # This fixes the dimension to 2

        with mp.Pool(nr_workers) as p:
            args_list = [(data[c == i],) for i in range(k)]
            results = p.starmap(recompute_centroids, args_list)

            for i, centroid in enumerate(results):
                centroids[i] = centroid
                    
        logging.debug(cluster_sizes)
        logging.debug(c)
        logging.debug(centroids)
    
    return total_variation, c


def computeClustering(args):
    if args.verbose:
        logging.basicConfig(format='# %(message)s',level=logging.INFO)
    if args.debug: 
        logging.basicConfig(format='# %(message)s',level=logging.DEBUG)
    
    X = generateData(args.samples, args.classes)


    avg_time = []
    cores = [1,2,4,8,16,32]

    for c in cores:
        times = []
        for _ in range(3):

            start_time = time.time()
            #
            # Modify kmeans code to use args.worker parallel threads
            total_variation, assignment = kmeans(args.k_clusters, X, nr_iter = args.iterations, nr_workers = c)
            #
            #
            end_time = time.time()
            logging.info("Clustering complete in %3.2f [s]" % (end_time - start_time))
            #print(f"Total variation {total_variation}")

            if args.plot: # Assuming 2D data
                fig, axes = plt.subplots(nrows=1, ncols=1)
                axes.scatter(X[:, 0], X[:, 1], c=assignment, alpha=0.2)
                plt.title("k-means result")
                #plt.show()        
                fig.savefig(args.plot)
                plt.close(fig)
            
            total_time = end_time-start_time
            times.append(total_time)
        
        avg_time.append(np.mean(times))

    return avg_time

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Compute a k-means clustering.',
        epilog = 'Example: kmeans.py -v -k 4 --samples 10000 --classes 4 --plot result.png'
    )
    parser.add_argument('--workers', '-w',
                        default='1',
                        type = int,
                        help='Number of parallel processes to use (NOT IMPLEMENTED)')
    parser.add_argument('--k_clusters', '-k',
                        default='3',
                        type = int,
                        help='Number of clusters')
    parser.add_argument('--iterations', '-i',
                        default='100',
                        type = int,
                        help='Number of iterations in k-means')
    parser.add_argument('--samples', '-s',
                        default='50000',
                        type = int,
                        help='Number of samples to generate as input')
    parser.add_argument('--classes', '-c',
                        default='3',
                        type = int,
                        help='Number of classes to generate samples from')   
    parser.add_argument('--plot', '-p',
                        type = str,
                        help='Filename to plot the final result')   
    parser.add_argument('--verbose', '-v',
                        action='store_true',
                        help='Print verbose diagnostic output')
    parser.add_argument('--debug', '-d',
                        action='store_true',
                        help='Print debugging output')
    args = parser.parse_args()
    
    measured_time = computeClustering(args)
    measured_speedup = [measured_time[0]/elem for elem in measured_time]

    cores = [1,2,4,8,16,32]
    theoretical_speedup = [1/((1-0.99)+(0.99/c)) for c in cores]

    amdahls_plot(measured_speedup, theoretical_speedup, cores, save=True)