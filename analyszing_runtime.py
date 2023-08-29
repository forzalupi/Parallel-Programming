#!/usr/bin/env python3

import multiprocessing
import random
import argparse
from math import pi
import time
import numpy as np
from problem_2_2 import amdahls_plot

def sample_pi(n):
    """Perform n steps of the Monte Carlo simulation for estimating Pi/4
    Returns the number of successes."""
    random.seed()
    s = 0
    for i in range(n):
        x = random.random()
        y = random.random()
        if x**2 + y**2 <= 1.0:
            s += 1
    return s

def compute_pi(n, num_workers, output=False):
    # Start overall time
    start_overall = time.time()

    m = n//num_workers

    # Start Parallel section
    start_para = time.time()

    with multiprocessing.Pool(num_workers) as p:
        s = p.map(sample_pi, [m]*num_workers)

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
    parser.add_argument('--steps','-s',default=100000,type=int,
                        help='Number of steps in the Monte Carlo simulation')
    args = parser.parse_args()

    cores = [1,2,4,8,16,32]
    # Create lists to store avg time and proportion over runs
    avg_time = []
    para_props = []

    for c in cores:
        times = []
        for _ in range(100):
            overall_time, parallel_proportion = compute_pi(args.steps, c)
            # Append overall time for each loop
            times.append(overall_time)

            # Save parallel proportion when core = 1
            if c == 1:
                para_props.append(parallel_proportion)
        
        # Save average time for every core
        avg_time.append(np.average(times))

    # Compute average proportion
    avg_p = np.average(para_props)

    # Compute the measured speedup for each core
    measured_speedup = [avg_time[0]/elem for elem in avg_time]
    # Compute the theoretivcal speedup for each core
    theoretical_speedup = [1/((1-avg_p)+(avg_p/c)) for c in cores]

    # Plot
    amdahls_plot(measured_speedup, theoretical_speedup, cores, save=True)