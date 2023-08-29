import math
import random
import numpy as np

import findspark
findspark.init()
from pyspark import SparkContext
import argparse

import time


def func(x):
    """
    f(x) = (((x*0xbc164501) & 0xffffffff) >> (32-l)) & 0xffffffff
    """
    l = int(math.log2(broad_m.value))
    return (((x*0xbc164501) & 0xffffffff) >> (32-l)) & 0xffffffff

def tab_hash8(key):
    """
    Tabulation hashing, 8-bit alphabet.
    """
    h_value = 0

    for i in range(8):
        chunk = (key >> (8*i)) & 0xff
        h_value ^= broad_table_8.value[i][chunk]

    return h_value

def leading_zero_count(x):
    """
    Count the leading zeroes of x.
    """
    count = 0
    while x & (1 << (31 - count)) == 0:
        count += 1
    return count

def worker_hll(elem):
    """
    Calculate the register index and nr of leading zeroes for each element in the RDD.
    Returns a tuple with reg_index as key and leading zeros as value.
    """        
    h_value = tab_hash8(elem)
    reg_index = func(h_value)
    leading_zeros = leading_zero_count(h_value) + 1
    
    return (reg_index, leading_zeros)

def para_hll(data):
    """
    HyperLogLog parallelized implemenation.
    """
    # Create a RDD with nr of workers chunks of data, apply the worker_hll function to each chunk
    worker_registers = sc.parallelize(data).map(worker_hll)

    # Max leading zeros for each register index
    index_max_val = worker_registers.reduceByKey(lambda a,b: max(a,b))

    # Collect
    registers_list = index_max_val.values().collect()
    # Check how many registers are empty
    V = broad_m.value-index_max_val.count()
    # Cast to array
    registers = np.array(registers_list, dtype=float)
    zeros = np.zeros(V, dtype=float)
    registers = np.concatenate((registers, zeros))

    # Estimated cardinality
    alpha = 0.7213/(1+(1.079/broad_m.value))
    est_cardinality = alpha*(m**2)/(np.sum(2**(-registers)))

    # Check for small or large cardinalities
    if est_cardinality <= ((5/2)*broad_m.value) and V > 0:
        est_cardinality = broad_m.value*math.log(broad_m.value/V)
    elif est_cardinality > ((1/30)*2**32):
        est_cardinality = (-2**32)*math.log(1-(est_cardinality/(2**32)))
    
    return est_cardinality


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = \
                                    'HyperLogLog')
    parser.add_argument('--workers','-w', default=1,type=int,
                        help='Number of parallel processes')
    args = parser.parse_args()

    cores = [1,2,4,8,16,24]

    avg_times = []

    for c in cores:
        times = []
        for i in range(10):
            sc = SparkContext(master = f"local[{c}]")

            # Look up table for 8-bit alphabet
            table_8 = [[random.getrandbits(32) for _ in range(2**8)] for _ in range(8)]
            # Create broadcast variable of table
            broad_table_8 = sc.broadcast(table_8)

            # Create data
            num_values = 1000000
            data = [random.getrandbits(64) for _ in range(num_values)]

            # Nr of registers
            m = 2**12
            broad_m = sc.broadcast(m)
            
            start = time.time()
            print(f"Estimated cardinality is: {para_hll(data)}")
            end = time.time()

            times.append(end-start)

            sc.stop()
        avg_times.append(sum(times)/10)

        
    print(avg_times)