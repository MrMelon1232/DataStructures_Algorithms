"""
We're going to explore binary search
the demo will be done in python, familiarizing the notion of these data structures through python
"""

# Time complexity: O(log n) | Space complexity: O(1)
def binary_search(arr, target):
    n = len(arr)
    r = n - 1
    l = 0
    
    while l <= r:
        # There are 2 formulas we can use, the classic one and the one that avoids int overflow
        # Classic one: (r + l) // 2 || clever one: l + ((r-l) // 2)
        m = l + ((r-l) // 2)

        if target == arr[m]:
            return True
        elif target > arr[m]:
            l = m + 1
        else:
            r = m - 1
    
    return False

# Testing
test = [-3, 0, 9, 5, 3, 7, 19]
print(binary_search(test, 5)) # Should print true
print(binary_search(test, 1000)) # Should print false


# Binary search for true/false condition based | finds the first occurence of when true value starts
def binary_search_condition(arr):
    n = len(arr)
    r = n - 1
    l = 0

    while l < r:
        m = l + ((r-l) // 2)
        print(m)

        if(arr[m]):
            r = m
        else:
            l = m + 1

    return l

# Testing 
B = [False, False, False, False, False, True, True, True, True]
print(binary_search_condition(B))