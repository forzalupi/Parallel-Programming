from pybloom_live import BloomFilter
import random
import string
import time

start = time.time()

# Read in data
with open('/data/2023-DAT470-DIT065/docs_nytimes_a.txt', 'r') as file:
    set_a = set()
    for line in file:
        # Split the document
        temp_list = line.split()
        # Make the word indices into ints
        temp_set = set(map(int, temp_list))
        # Add to set A
        set_a |= temp_set

with open('/data/2023-DAT470-DIT065/docs_nytimes_b.txt', 'r') as file:
    set_b = set()
    for line in file:
        # Split the document
        temp_list = line.split()
        # Make the word indices into ints
        temp_set = set(map(int, temp_list))
        # Add to set A
        set_b |= temp_set

# Set p and the length of the universe
p = 0.01
len_U = 102660

# Construct the filters
filter_A = BloomFilter(capacity=len_U, error_rate=p)
for x in set_a:
        filter_A.add(x)
filter_B = BloomFilter(capacity=len_U, error_rate=p)
for x in set_b:
        filter_B.add(x)


# Create test strings
num_strings = 100000
test_strings = set()
while len(test_strings) < num_strings:
    test_string = ''.join(random.choices(string.ascii_lowercase, k=10))
    if test_string not in filter_A or test_string not in filter_B:
        test_strings.add(test_string)

len_test_strings = len(test_strings)

# Test for false positives
fp_a = 0
fp_b = 0
for test_string in test_strings:
    if test_string in filter_A:
        fp_a += 1
    if test_string in filter_B:
        fp_b += 1

print(f"""FP rate of Bloom filter A: {fp_a/len_test_strings},
FP rate of Bloom filter B: {fp_b/len_test_strings}""")

end = time.time()

print(end-start)