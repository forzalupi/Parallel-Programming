import random

random.seed(1337)

# Look up table for 8-bit alphabet
table_8 = [[random.getrandbits(32) for _ in range(2**8)] for _ in range(8)]
# Look up table for 16-bit alphabet
table_16 = [[random.getrandbits(32) for _ in range(2**16)] for _ in range(4)]

def tab_hash8(key, table):
    """
    Tabulation hashing, 8-bit alphabet.
    """
    h_value = 0

    for i in range(8):
        chunk = (key >> (8*i)) & 0xff
        h_value ^= table[i][chunk]

    return h_value

def tab_hash16(key, table):
    """
    Tabulation hashing, 16-bit alphabet.
    """
    h_value = 0
    
    for i in range(4):
        chunk = (key >> (4*i)) & 0xffff
        h_value ^= table[i][chunk]

    return h_value


key_64 = 0x123456789abcdef

hash_8 = tab_hash8(key_64, table_8)
hash_16 = tab_hash16(key_64, table_16)