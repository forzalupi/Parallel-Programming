#!/usr/bin/env python3

import sys
from collections import defaultdict
import math

count = defaultdict(int)

def mem_approx(my_dict):
    """
    Approximate memory needed by an efficient implementation.
    """
    
    n = len(my_dict)
    num_cells = 2**(math.ceil(math.log2(n)))

    # Empty cells
    empty_cell_bytes = (num_cells-n)*4
    # Value
    value_bytes = n*4
    # Keys
    key_bytes = sum(2*(len(k)+1) for k in my_dict.keys())
    # Pointers
    pointer_bytes = num_cells*8

    # Sum
    total_bytes = empty_cell_bytes+value_bytes+key_bytes+pointer_bytes

    return total_bytes

if __name__ == '__main__':
    for line in sys.stdin:
        for word in line.strip().split():
            count[word] += 1

    """for k,v in count.items():
        print(k,v)"""
    
    print(f"Size of the data structure in bytes: {mem_approx(count)}")

    """# Testing
    my_dict = {"hej": 1, "hejsan": 1, "qejsan": 1}

    print(mem_approx(my_dict))"""