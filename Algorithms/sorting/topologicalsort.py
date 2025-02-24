"""
We're going to explore topological sort using dfs and bfs
the demo will be done in python, familiarizing the notion of these data structures through python
"""

# Topological sort is an algorithm used to sort nodes in directed acyclic graphs (DAG) in linear order such that vertice A comes before vertice B if node A goes to B
# usually used for problems like course scheduling, where courses have pre-requisites
# can help detect unwanted cycles in DAGs

# There are two methods to implement toplogical sort, using either DFS or BFS
# 1. DFS: The dfs version of this algorithm does not necessary sort it in the exact same order every time. Order may be non-unique
# 2. BFS (Kahn's algorithm): The bfs version of this algorithm has a unique solution
# Time Complexity: O(V + E) | Space Complexity: O(V) | Where V = vertices, E = edges

# Method 1: DFS
def topologicalSortUtil(v, adj, visited, stack):
    
    # Mark the current node as visited
    visited[v] = True

    # Recur for all adjacent vertices
    for i in adj[v]:
        if not visited[i]:
            topologicalSortUtil(i, adj, visited, stack)
    
    # Push current vertex to stack which stores the result
    stack.append(v)

# Function to perform Topological Sort using DFS
def topologicalSortDFS(adj, V):

    # Stack to store the results
    stack = []

    # Create our visited array
    visited = [False] * V

    # Call the recursive helper function to store
    # Topological Sort starting from all vertices one by one
    for i in range(V):
        if not visited[i]:
            topologicalSortUtil(i, adj, visited, stack)
    
    # Print contents of stack
    print("Topological sorting of the graph:", end=" ")
    while stack:
        print(stack.pop(), end=" ")
    

# Num of nodes
V = 4

# Edges
edges = [[0, 1], [1, 2], [3, 1], [3, 2]] 

# Adjacency List for testing
from collections import defaultdict
D = defaultdict(list)
for u, v in edges:
    D[u].append(v)

topologicalSortDFS(D, V)


# Method 2: BFS 
# Steps:
# 1. Create indegree array storing all indegree of each nodes
# 2. Store nodes with indegree = 0 in our queue
# 3. Create the queue loop to loop through our items in our queue
# 4. Pop the first item and store it in our result array ; decrement the indegree of its adjacent nodes
# 5. Store nodes with indegree = 0 in our queue
# 6. If the length of our result is not equal to the number of nodes, then theres a cycle
# NOTE: we can create our indegree array and fill it when creating our adjacency matrix

from collections import deque

def topologicalSortBFS(adj, V):
    # Vector to store indegree of each vertex (how many nodes point to it)
    indegree = [0] * V
    
    # loop to calculate the indegree
    for i in range(V):
        for vertex in adj[i]:
            indegree[vertex] += 1
    
    # Queue to store vertices with indegree 0 (starting nodes that have no pre-reqs)
    q = deque()
    for i in range(V):
        if indegree[i] == 0:
            q.append(i)
    
    result = []

    while q:
        node = q.popleft()
        result.append(node)

        # Decrease indegree of adjacent vertices as the current node is in topological order 
        for adjacent in adj[node]:
            indegree[adjacent] -= 1

            # If indegree becomes 0, push it to the queue
            if indegree[adjacent] == 0:
                q.append(adjacent)
        
    # Check for a cycle
    if len(result) != V:
        print("Graph contains cycle!")
        return []
    return result
            

# Number of nodes
n = 6

# Edges 
edges = [[0, 1], [1, 2], [2, 3], [4, 5], [5, 1], [5, 2]]

C = defaultdict(list)
for u, v in edges:
    C[u].append(v)

r2 = []
r2 = topologicalSortBFS(C, n)
print(r2)