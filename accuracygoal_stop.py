#!/usr/bin/env python3

import multiprocessing
import numpy as np
import random
import argparse
from math import pi
import time
from amdahl import amdahls_plot

def sample_pi(q, n, seed):
    """Perform n steps of the Monte Carlo simulation for estimating Pi/4
    Returns the number of successes and the total number of steps."""
    random.seed(seed)
    s = 0
    for i in range(n):
        x = random.random()
        y = random.random()
        if x**2 + y**2 <= 1.0:
            s += 1
    # Put result in queue
    q.put((s, n))

def compute_pi(accuracy, num_workers, explicit_seed, batch_size, output=False):
    # Start overall time
    start_overall = time.time()

    # Initialize queue and variables
    q = multiprocessing.Queue()
    n_total = 0
    s_total = 0
    i = 0

    error = 0
    pi_est = 0 

    while True:
        # Create variables for error and pi_est
        error = 0
        pi_est = 0

        # Create list of seeds
        seeds = [explicit_seed + i*num_workers + j for j in range(num_workers)]

        # Create list of workers
        workers = [multiprocessing.Process(target=sample_pi, args=(q, batch_size, seeds[j])) for j in range(num_workers)]
        
        # Run all the workers
        for w in workers:
            w.start()
        
        # Join workers when all are ready
        for w in workers:
            w.join() 

        # Get the results from the queue for all the workers
        for _ in range(num_workers):
            s, n = q.get()
            n_total += n
            s_total += s

        # Calculate the estimated pi and error
        pi_est = 4.0 * s_total / n_total
        error = abs(pi - pi_est)

        # Break if accuracy goal is met
        if error < accuracy:
            break
        
        # Add +1 for each iteration
        i += 1

        if output:
            # Print each loop
            print(f"""samples: {n_total},
            s-total: {s_total},
            estimated pi: {pi_est},
            error: {error}""")

    # End overall time
    end_overall = time.time()

    # Compute the total time
    overall_time = end_overall - start_overall

    # Samples per second
    samples_per_second = n_total / overall_time

    if output:
        # Print result
        print(f"""samples: {n_total},
        s-total: {s_total},
        estimated pi: {pi_est},
        error: {error},
        samples/s: {samples_per_second}""")

    return samples_per_second

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = \
                                    'Compute Pi using Monte Carlo simulation.')
    parser.add_argument('--workers','-w',default=1,type=int,
                        help='Number of parallel processes')
    parser.add_argument('--accuracy','-a',default=0.001,type=float,
                        help='Accuracy goal for approximating pi')
    parser.add_argument('--seed', '-d', default=1337,type=int,
                        help='Seed for the Monte Carlo simulation')
    parser.add_argument('--batch_size', '-b', default=10000,type=int,
                        help='Number of points per worker')
    args = parser.parse_args()
    
    cores = [1,2,4,8,16,32]
    average_samples = []

    # Loop over all cores
    for c in cores:
        samples = []
        # Average over 5 iterations
        for _ in range(5):
            samples_per_second= compute_pi(args.accuracy, c, args.seed, args.batch_size, output=False)
            samples.append(samples_per_second)

        average_samples.append(np.mean(samples))

    # Calculate the speedup based on samples/s
    measured_speedup = [elem/average_samples[0] for elem in average_samples]

    print("All done")
    amdahls_plot(measured_speedup, cores, save=True)