"""
We're going to explore arrays, dynamic arrays, static arrays and strings
the demo will be done in python, familiarizing the notion of these data structures through python
Note that in python, arrays are also called lists
"""
# Arrays | note that arrays are mutable (which means they can be modified, unlike tuples)
A = [1, 2, 3]

# Append - Insert element at the end of an array - On average: O(1)
A.append(5)
print(A) # Should print [1, 2, 3, 5]

# Pop - Removes element at the end of an array - On average: O(1)
A.pop()
print(A) # Should print [1, 2, 3]

# Modify an element - O(1)
A[0] = 0
print(A) # Should print [0, 2, 3]

# Accessing element given index i - O(1)
print(A[0]) # Should print 0

# Finding specific element in array - O(n)
if 0 in A:
    print(True)

# Length of array - O(1)
print(len(A)) # Should print 3


"""
Here is the following time complexity of static and dynamic arrays:

Appending to end        | *O(1) ('*' means on average)
Popping from end        | O(1)
Insertion not from end  | O(n)
Deletion  not from end  | O(n)
Modifying element       | O(1)
Random access           | O(1)
Checking element exists | O(n)
"""


###################################################################
# Strings

# Append to end of string - O(n)
s = "hello"
b = s + 'z'
print(b)

# Find something in strnig - O(n)
if 'f' in s:
    print(True)

# Length of string - O(1)
print(len(s)) # Should print 5


"""
Here is the following time complexity of strings (note that in python, strings are immutable):

Appending to end        | O(n) 
Popping from end        | O(n)
Insertion not from end  | O(n)
Deletion  not from end  | O(n)
Modifying element       | O(n)
Random access           | O(1)
Checking element exists | O(n)
"""