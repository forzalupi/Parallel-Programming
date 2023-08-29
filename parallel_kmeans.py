#!/usr/bin/env python3

import multiprocessing
import random
import argparse
from math import pi
import time
import numpy as np

def sample_pi(args):
    """Perform n steps of the Monte Carlo simulation for estimating Pi/4
    Returns the number of successes."""
    n, seed = args
    random.seed(seed)
    s = 0
    for i in range(n):
        x = random.random()
        y = random.random()
        if x**2 + y**2 <= 1.0:
            s += 1
    return s

def compute_pi(n, num_workers, explicit_seed, output=False):
    # Start overall time
    start_overall = time.time()

    m = n//num_workers
    # Create list of seeds
    seeds = [explicit_seed+i for i in range(num_workers)]

    # Start Parallel section
    start_para = time.time()

    with multiprocessing.Pool(num_workers) as p:
        s = p.map(sample_pi, zip([m]*num_workers, seeds))

    # End parallel section
    end_para = time.time()

    n_total = m*num_workers
    s_total = sum(s)
    pi_est = 4.0 * s_total / n_total
    print('Steps\t# Succ.\tPi est.\tError')
    print(f'{n_total:6d}\t{s_total:6d}\t{pi_est:1.5f}\t{pi-pi_est:1.5f}')

    # End overall time
    end_overall = time.time()

    # Compute the total time
    overall_time = end_overall - start_overall
    # Compute the parallel section
    parallel_section = end_para - start_para

    parallel_proportion = parallel_section/overall_time

    if output:
        print(f"""Overall time: {overall_time},
        Parallel section: {parallel_section},
        Serial section: {overall_time-parallel_section},
        Proportion of serial computation: {(overall_time-parallel_section)/overall_time}""")
    
    return overall_time, parallel_proportion

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = \
                                    'Compute Pi using Monte Carlo simulation.')
    parser.add_argument('--workers','-w',default=1,type=int,
                        help='Number of parallel processes')
    parser.add_argument('--steps','-s',default=1000,type=int,
                        help='Number of steps in the Monte Carlo simulation')
    parser.add_argument('--seed', '-d', default=1337,type=int,
                        help='Seed for the Monte Carlo simulation')
    args = parser.parse_args()
    compute_pi(args.steps, args.workers, args.seed)