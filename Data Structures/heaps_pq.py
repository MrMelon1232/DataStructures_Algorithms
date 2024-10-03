"""
We're going to explore heaps and priority queues
the demo will be done in python, familiarizing the notion of these data structures through python
"""

# Building a Min Heap (Heapify)
# time compplexity: O(n) | space complexity: O(1)

A = [-1, 4, 2, 1, 0, 5, 9, 18, -4]

import heapq
heapq.heapify(A)
print(A)

# Inserting element - Heap Push 
# time complexity: O(log n)
heapq.heappush(A, 20)
print(A)

# Removing element (min value since its min heap) - Heap pop
# time complexity: O(log n)
heapq.heappop(A)
print(A)

# Heap sort | use a heap and keep popping min to get sorted array
# time complexity: O(n log n) | space complexity: O(n)
def heapsort(arr):
    heapq.heapify(arr)
    n = len(arr)
    new_list = [0] * n

    for i in range(n):
        min = heapq.heappop(arr)
        new_list[i] = min
    
    return new_list

print(heapsort([-4, 3, 3, 10, 2, 8, 4, -10]))

# Heap push pop | pushes and pops
# time complexity: O(log n)
heapq.heappushpop(A, 99)
print(A)

# Peek | first item is always min
# time complexity: O(1)
A[0]


# Max heap | make the numbers in the array negative and make a min heap
B = [-4, 3, 1, 10, -30, 8, 4, 2]
n = len(B)
for i in range(n):
    B[i] = -B[i]

heapq.heapify(B)
print(B)

# to retrieve largest number
largest = -heapq.heappop(B)
print(largest)

# inserting - make it negative
heapq.heappush(A, -7) # pushing 7 to max heap


# Build heap from scratch | O(n log n)
C = [-5, 4, 0, 2, 8, 5]
heap = []

for x in C:
    heapq.heappush(heap, x)
    print(heap)


