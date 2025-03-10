"""
We're going to explore the Union Find algorithm used to solve connected graph problems that involve merging sets, checking if elements are connected
It involves graphs but not necessarily connected ones. Essentially helps us group nodes/items together
the demo will be done in python, familiarizing the notion of these data structures through python
"""

# Union find is used to group sets/nodes based on a metric. We can also find which group a node belongs to as well. 
# USED FOR:
#   1. Finding connected components in graph
#   2. Merging sets or determining if 2 elements are in the same set
# General Process and Logic
#   1. Connect our nodes that we want to group together based on a metric (can be a representative element) | when we use the find function, it will return the representative element
#   2. Make our representative element the root 
#   3. To merge 2 unions, set the root of a union as a child of another


'''
We will define 2 functions:
    1. Find: finds the root parent of the current node
    2. Union: union 2 different nodes together

'''

# Time Complexity: O(log n)
# Space Complexity: O(n)



