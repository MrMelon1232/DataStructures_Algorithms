"""
We're going to explore bubble sort
the demo will be done in python, familiarizing the notion of these data structures through python
"""

# Bubble sort
# the biggest number will "bubble" to the end of the list with every iteration
# time complexity: O(n^2) | space complexity: O(1)
def bubblesort(arr):
    n = len(arr)
    flag = True
    while flag:
        flag = False
        for i in range(1, n):
            if arr[i-1] > arr[i]:
                arr[i-1], arr[i] = arr[i], arr[i-1]
                flag = True


# testing
A = [-5, 2, 3, -6, 9, 10, 4]
bubblesort(A)
print(A)


"""
We're going to explore insertion sort
the demo will be done in python, familiarizing the notion of these data structures through python
"""

# Insertion sort
# comapare the values from index j and j-1 where j starts at i while decrementing j until we reach the first item in the list
# continue to increment i afterwards
# TLDR: compare current index item with previous items of list to see if its smaller, if so move it back and compare again
# time complexity: O(n^2) | space complexity: O(1)
def insertionsort(arr):
    n = len(arr)
    
    for i in range(1, n):
        for j in range(i, 0, -1):
            if arr[j] < arr[j-1]:
                arr[j-1], arr[j] = arr[j], arr[j-1]
            else:
                break


# testing
B = [-5, 2, 3, -6, 9, 10, 4]
insertionsort(B)
print(B)

"""
We're going to explore selection sort
the demo will be done in python, familiarizing the notion of these data structures through python
"""

# Selection sort
# compare indexed item with all items in list until we find minimum number out of all items
# swap indexed item with found minimum after
# time complexity: O(n^2) | space complexity: O(1)
def selectionsort(arr):
    n = len(arr)    

    for i in range(0, n):
        min = i
        for j in range(i+1, n):
            if arr[j] < arr[i]:
                min = j
        arr[i], arr[min] = arr[min], arr[i]



# testing
C = [-5, 2, 3, -6, 9, 10, 4]
selectionsort(C)
print(C)

"""
We're going to explore merge sort
the demo will be done in python, familiarizing the notion of these data structures through python
"""

# Merge sort
# divide and conquer
# divide arrays into sub arrays until we reach base case using recursion, then merge arrays by comparing items in both arrays
# time complexity: O(n log n) | space complexity: O(n)
def mergesort(arr):
    n = len(arr)    
 
    # base case
    if n == 1:
        return arr
    
    # Here generates a copy of arrays, hence O(n) space
    m = len(arr) // 2
    L = arr[:m]
    R = arr[m:]

    # Continuously divide array until base case
    L = mergesort(L)
    R = mergesort(R)
    l, r = 0, 0
    L_len = len(L)
    R_len = len(R)

    # Create index and sorted array holder
    sorted_arr = [0] * n
    i = 0

    # Loop while we still have numbers in either left or right subarrays
    while l < L_len and r < R_len:
        if L[l] < R[r]:
            sorted_arr[i] = L[l]
            l+=1
        else:
            sorted_arr[i] = R[r]
            r+=1
        i+=1
    
    # Add the rest of left array items
    while l < len(L):
        sorted_arr[i] = L[l]
        i+=1
        l+=1

    # Add the rest of right array items    
    while r < len(R):
        sorted_arr[i] = R[r]
        i+=1
        r+=1
    
    return sorted_arr

# testing
D = [-5, 2, 3, -6, 9, 10, 4]
sorted_arr = mergesort(D)
print(sorted_arr)


"""
We're going to explore quick sort
the demo will be done in python, familiarizing the notion of these data structures through python
"""

# Quick sort
# Choose a pivot and compare the items to the pivot 
# if item its smaller than the pivot put it in left array
# if item its bigger than the pivot put it in right 
# after splitted till end, all items are already sorted so just merge arrays
# note that: bad pivot = bad time complexity | usually choose last item as pivot
# time complexity: O(n log n) | space complexity: O(n)
def quicksort(arr):
    if len(arr) <= 1:
        return arr
    
    pivot = arr[-1]

    L = [x for x in arr[:-1] if x <= pivot]
    R = [x for x in arr[:-1] if x > pivot]

    L = quicksort(L)
    R = quicksort(R)

    return L + [pivot] + R

# testing
E = [-5, 2, 3, -6, 9, 10, 4]
sorted_arr = quicksort(E)
print(sorted_arr)


# Counting sort
# count the occurence of each number, note that large numbers will make this algorithm very slow and have a big time complexity
# using the new occurence array, loop through occurence array, making sure that all numbers have properly replaced items in problem array to sort it, increment problem array each time an item is replaced
# ex: if there are 1 '0' and 2 '1', then we replace first number in arr to '0', then 2nd to '1' then 3rd to '1' etc
# note that: only use this algo if 'k' is small, where k is the max number in the array
# note: this is written for positive arrays
# time complexity: O(k + n) | space complexity: O(k)
def counting_sort(arr):
    n = len(arr)
    maxx = max(arr)
    counts = [0] * (maxx + 1)

    # create our occurence array by incrementing the number of times each number repeats
    for x in arr:
        counts[x] += 1

    # create index for occurence array
    i = 0

    # loop over occurence array and change problem array
    for c in range(maxx + 1):
        while counts[c] > 0:
            arr[i] = c
            i += 1
            counts[c] -= 1

# testing
F = [1, 3, 3, 2, 2, 0, 1, 5, 4, 7, 9]
counting_sort(F)
print(F)


# Python sort algorithm, time sort
# time complexity: O(n log n) | space complexity: O(1)
G = [1, 3, 3, 2, 2, 0, 1, 5, 4, 7, 9]
G.sort()
print(G)