"""
We're going to explore binary trees
the demo will be done in python, familiarizing the notion of these data structures through python
"""

# Binary tree
class TreeNode:
    def __init__(self, val, left=None, right=None) -> None:
        self.val = val
        self.left = left
        self.right = right
    
    def __str__(self) -> str:
        return str(self.val)

# Create tree
A = TreeNode(1)
B = TreeNode(2)
C = TreeNode(3)
D = TreeNode(4)
E = TreeNode(5)
F = TreeNode(6)

A.left = B
A.right = C
B.left = D
B.right = E
C.left = F

# This is our tree
#       1
#   2       3
# 4   5   6    

# Note that trees can also be implemented with a dictionary where each key contains a dictionary pointing to its child nodes

# Recursive pre order traversal (DFS) | use this when the desired node is far
# Time complexity: O(n) | space complexity: O(n)
def pre_order(node):
    if not node:
        return
    
    print(node)
    pre_order(node.left)
    pre_order(node.right)

#testing
print("Listing pre_order traversal")
pre_order(A)

# DFS using stack
# Time complexity: O(n) | space complexity: O(n)
def dfs_stack(node):
    stack = [node]

    # loop while stack is not empty
    while stack:
        curr = stack.pop()
        print(curr)
        if(curr.right):
            stack.append(curr.right)
        if(curr.left):
            stack.append(curr.left)
        

#testing
print("Listing dfs_stack traversal")
dfs_stack(A)

# Recursive in order traversal (DFS)
# Time complexity: O(n) | space complexity: O(n)
def in_order(node):
    if not node:
        return
    
    in_order(node.left)
    print(node)
    in_order(node.right)

#testing
print("Listing in_order traversal")
in_order(A)

# Recursive post order traversal (DFS)
# Time complexity: O(n) | space complexity: O(n)
def post_order(node):
    if not node:
        return
    
    post_order(node.left)
    post_order(node.right)
    print(node)

#testing
print("Listing post_order traversal")
post_order(A)

# BFS traversal | level traversal | use this when we're looking for close nodes
# Time complexity: O(n) | Space complexity: O(n)
from collections import deque

def bfs(node):
    q = deque()
    q.append(node)

    while q:
        curr = q.popleft()
        print(curr)
        if(curr.left):
            q.append(curr.left)
        if(curr.right):
            q.append(curr.right)

#testing
print("Listing bfs traversal")
bfs(A)


# Using DFS, we will search for a value in the node
def search(node, target):
    if not node:
        return False

    if node.val == target:
        return True

    return search(node.left, target) or search(node.right, target)

#testing
print("Listing search dfs method")
print(search(A, 5)) # Should print true
print(search(A, 10)) # Should print false



# Binary search trees (BSTs)
# note that binary search trees are considered balanced when the left node from the root is smaller than the root and the right node is bigger than the root node for all subtrees

#           5
#       1       8
#   -1    3   7   9

A2 = TreeNode(5)
B2 = TreeNode(1)
C2 = TreeNode(8)
D2 = TreeNode(-1)
E2 = TreeNode(3)
F2 = TreeNode(7)
G2 = TreeNode(9)

A2.left, A2.right = B2, C2
B2.left, B2.right = D2, E2
C2.left, C2.right = F2, G2

# testing
print("Listing in order for a BST")
in_order(A2)

# searching
# time complexity: O(log n) (if tree is balanced) | space complexity: 0(log n)
def search_bst(node, target):
    if not node:
        return
    
    if node.val == target:
        return True
    
    # if target is smaller than current value than it has to be in the left subtree
    if(target < node.val):
        return search_bst(node.left, target)
    
    # if target is bigger than current value than it has to be in the right subtree
    if(target > node.val):
        return search_bst(node.right, target)