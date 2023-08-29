import math
import random
import numpy as np
import time

def func(x, m):
    """
    f(x) = (((x*0xbc164501) & 0xffffffff) >> (32-l)) & 0xffffffff
    """
    l = int(math.log2(m))
    return (((x*0xbc164501) & 0xffffffff) >> (32-l)) & 0xffffffff

def tab_hash8(key, table):
    """
    Tabulation hashing, 8-bit alphabet.
    """
    h_value = 0

    for i in range(8):
        chunk = (key >> (8*i)) & 0xff
        h_value ^= table[i][chunk]

    return h_value

def leading_zero_count(x):
    """
    Count the leading zeroes of x.
    """
    count = 0
    while x & (1 << (31 - count)) == 0:
        count += 1
    return count

def hll(data, m, table):
    """
    HyperLogLog implemenation.
    """

    registers = np.zeros(m, dtype=np.uint32)

    for elem in data:
        h_value = tab_hash8(elem, table)
        reg_index = func(h_value, m)
        leading_zeroes = leading_zero_count(h_value) + 1

        registers[reg_index] = max(registers[reg_index], leading_zeroes)
    

    # Estimated cardinality
    alpha = 0.7213/(1+(1.079/m))
    est_cardinality = alpha*(m**2)/(np.sum(2**(-registers)))

    # Check how many registers are empty
    V = 0
    for i in registers:
        if i == 0:
            V += 1

    # Check for small or large cardinalities
    if est_cardinality <= ((5/2)*m) and V > 0:
        est_cardinality = m*math.log(m/V)
    elif est_cardinality > ((1/30)*2**32):
        est_cardinality = (-2**32)*math.log(1-(est_cardinality/(2**32)))
    
    return est_cardinality


if __name__ == "__main__":
    # Look up table for 8-bit alphabet
    table_8 = [[random.getrandbits(32) for _ in range(2**8)] for _ in range(8)]

    # Create data
    num_values = 1000000
    data = [random.getrandbits(64) for _ in range(num_values)]

    # Nr of registers
    m = 2**20

    start = time.time()
    print(f"Estimated cardinality is: {hll(data, m, table_8)}")
    end = time.time()

    print(f"time: {end-start}")