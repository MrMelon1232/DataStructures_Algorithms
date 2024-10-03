"""
We're going to explore graphs
the demo will be done in python, familiarizing the notion of these data structures through python
"""

# Graphs | note that trees are also graphs but not every graph is a tree
# Trees must be acyclic (no cycle) and connected
# Graphs are represented using edges (connections between vertices) and vertices (vertices = nodes)

# Method 1 of representing graph -  Array of edges
n = 8
A = [[0, 1], [1, 2], [0, 3], [3, 4], [3, 6], [3, 7], [4, 2], [4, 5], [5, 2]]

# Method 2 of representing graph - Adjacency Matrix
# each row represents the first element, and the columns represent the connection 
# ex: [0, 4] --> means that in row 0 there is element at index 4 toggled as 1
M = []
for i in range(n):
    M.append([0] * n)
print(M)

for u, v in A:
    M[u][v] = 1

print(M)


# Method 3 of representing graph - Adjacency list
# prefered method for me 
from collections import defaultdict

D = defaultdict(list)
for u, v in A:
    D[u].append(v)
print(D)

# Accessing connections
print(D[3])


# DFS with recursion for graphs - O(v + e)
seen = set()
source_node = 0
seen.add(source_node)
def DFS_recursion(node):
    print(node)
    for n in D[node]:
        if n not in seen:
            seen.add(n)
            DFS_recursion(n)

DFS_recursion(source_node)

# DFS with stack for graphs - O(v + e)
source = 0
seen = set()
seen.add(source)
stack = [source]

def DFS_stack():
    while stack:
        node = stack.pop()
        print(node)
        for n in D[node]:
            if n not in seen:
                seen.add(n)
                stack.append(n)

DFS_stack()

    

# BFS for graphs
from collections import deque

source = 0
seen = set()
seen.add(source)
queue = deque()
queue.append(source)

def BFS():
    while queue:
        node = queue.popleft()
        print(node)
        for n in D[node]:
            if n not in seen:
                seen.add(n)
                queue.append(n)

BFS()


# Creation of graph
class Node:
    def __init__(self, val) -> None:
        self.val = val
        self.neighbors = []

A = Node('A')
B = Node('B')
C = Node('C')
D = Node('D')

A.neighbors.append(B)
B.neighbors.append(A)

C.neighbors.append(D)
C.neighbors.append(C)