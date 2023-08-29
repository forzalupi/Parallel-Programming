def murmur3_32(key, seed=1337):
    """
    Murmurhash implementation.
    """
    # Assign variables
    h = seed
    c1 = 0xcc9e2d51
    c2 = 0x1b873593
    r1 = 15
    r2 = 13
    m = 5
    n = 0xe6546b64
    
    length = len(key)

    # Split into four byte chunks, perform bitwise operations
    for i in range(0, length // 4):
        k = int.from_bytes(key[i*4:i*4+4], byteorder='little', signed=False)
        k = (k * c1) & 0xffffffff
        k = ((k << r1) | (k >> (32 - r1))) & 0xffffffff
        k = (k * c2) & 0xffffffff
        h = (h ^ k) & 0xffffffff
        h = ((h << r2) | (h >> (32 - r2))) & 0xffffffff
        h = ((h * m) + n) & 0xffffffff

    # Remaining bytes
    remaining_chunk = key[length-(length%4):].ljust(4, b'\0')
    k = int.from_bytes(remaining_chunk, byteorder="little", signed=False)
    k &= 0xffffffff
    k = (k * c1) & 0xffffffff
    k = ((k << r1) | (k >> (32 - r1))) & 0xffffffff
    k = (k * c2) & 0xffffffff
    h = (h ^ k) & 0xffffffff

    # Finalize.
    h = (h ^ length) & 0xffffffff
    h ^= h >> 16
    h = (h * 0x85ebca6b) & 0xffffffff
    h ^= h >> 13
    h = (h * 0xc2b2ae35) & 0xffffffff
    h ^= h >> 16

    return h


def string_encoder(strings):
    """
    """
    encoded_strings = [s.encode() for s in strings]

    return encoded_strings

# Test strings
strings = ["h", "he", "hel", "hell", "hello", "hello ", "hello w", "hello wo", "hello wor", "hello worl", "hello world"]
encoded_strings = string_encoder(strings)

for string in encoded_strings:
    hash_value = murmur3_32(string, 1234)

    print("Input string: {}".format(string))
    print("Hash value: {}".format(hash_value))
    print("Hash value: 0x{:08x}".format(hash_value))
    print("----")