#!/usr/bin/env python3

import sys

class TrieNode:
    def __init__(self, word = '', count = None):
        self._children = list()
        self._child_chars = list()
        self._num_children = 0
        if len(word) == 0:
            self._count = 1 if count is None else count
        else:
            self._count = 0
            try:
                i = self._child_chars.index(word[0])
                self._children[i] = TrieNode(word[1:],count)
            except ValueError:
                self._child_chars.append(word[0])
                self._num_children += 1
                self._children.append(TrieNode(word[1:],count))

    def add(self, word, count=None):
        if len(word) == 0:
            self._count += 1 if count is None else count
        else:
            if word[0] not in self._child_chars:
                self._child_chars.append(word[0])
                self._children.append(TrieNode(word[1:], count))
                self._num_children += 1
            else:
                i = self._child_chars.index(word[0])
                self._children[i].add(word[1:], count)

    def __iter__(self):
        for i in range(self._num_children):
            for k,v in self._children[i]:
                if v > 0: yield (self._child_chars[i]+k, v)
        if self._count > 0: yield ('', self._count)

    def __getitem__(self, key):
        if len(key) == 0 and self._count > 0:
            return self._count
        if len(key) == 0: raise KeyError('key not found')
        try:
            i = self._child_chars.index(key[0])
        except ValueError:
            raise KeyError('key not found')
        return self._children[i][key[1:]]

def count_nodes(node):
    """
    """
    if node is None:
        return 0
    count = 1
    for child in node._children:
        count += count_nodes(child)
    return count


def mem_approx(trie):
        """
        Approximate memory needed by an efficient implementation.
        """
        # Call the recursive function counting nodes
        num_nodes = count_nodes(trie)
        num_children = num_nodes-1
        # Bytes
        node_bytes = 21*num_nodes
        pointer_bytes = 8*num_children
        char_bytes = 2*num_children

        # Sum
        total_bytes = node_bytes+pointer_bytes+char_bytes

        return total_bytes

if __name__ == '__main__':
    trie = TrieNode('',0)
    for line in sys.stdin:
        for word in line.strip().split():
            trie.add(word)

    """for k,v in trie:
        print(k,v)"""

    """# Testing
    my_l = ["hej", "hejsan", "qejsan"]
    for w in my_l:
        trie.add(w)"""
    
    print(f"Size of the data structure in bytes: {mem_approx(trie)}")