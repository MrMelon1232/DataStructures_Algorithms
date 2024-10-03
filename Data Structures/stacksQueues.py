"""
We're going to explore stacks and queues
the demo will be done in python, familiarizing the notion of these data structures through python
"""

# Stack (LIFO) | stacks can be implemented using an array | Note that stacks in python are in reversed order, meaning the last item is the top of the stack
stack = []
print(stack)

# Append to top of stack - O(1)
stack.append(1)
stack.append(2)
stack.append(3)
print(stack) # Should print [1, 2, 3]

# Pop from stack - O(1)
x = stack.pop()
print(x) # Should print 3
print(stack) # Should print [1, 2]

# Peek from stack - O(1)
print(stack[-1]) # Should print 2

# Ask if something is in the stack - O(1)
if stack:
    print(True)


###################################################################
# Queues (FIFO) | queues can be implemented using a doubly linked list or using deque from collections
from collections import deque

q = deque()
print(q)

# Enqueue - Add element to the right - O(1)
q.append(5)
q.append(6)
print(q) # Should print [5,6]

# Dequeue - Remove element from the left - O(1)
q.popleft()
print(q) # Should print [6]

# Peek from left side - O(1)
q[0]
