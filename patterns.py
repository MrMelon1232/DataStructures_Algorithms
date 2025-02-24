"""
We're going to explore common patterns in leetcode style questions. Review this for interviews!
the demo will be done in python, familiarizing the notion of these data structures through python
"""


# Pattern 1: PREFIX SUM
# KEY WORDS: QUERY SUM OF ELEMENTS IN SUBARRAY
# At each index of our array, the value represents the sum of elements until index i
# For Sum(i, j) = prefix[j] - prefix[i-1] | helps us find any prefix sum
# Time Complexity: O(n) | Space Complexity: O(n) or O(1) if you modify input arr
def prefix(nums):
    
    n = len(nums)
    prefix_sum = [0] * n
    prefix_sum[0] = nums[0]
    
    # fill our prefix sum array
    for i in range(1, n):
        prefix_sum[i] = nums[i] + prefix_sum[i-1]

    # lets say we want to find the sum from index 2 to 4
    res = prefix[4] - prefix[2-1]

    # we can also modify our input array if allowed directly
    for i in range(1, n):
        nums[i] += nums[i-1]


# Pattern 2: 2 POINTERS
# Use a start and end pointer and squeeze our array until we find our answer
# Time Complexity: O(n) | Space Complexity: O(1)

# Example Problem: Palindrome, where a string is a palindrome if string is the same when order is reversed
def palindrome(str):

    # strip the string of all non numbers/alphabet symbols and spaces
    str = ''.join(char for char in str if char.isalnum()).lower()

    start = 0
    end = len(str)

    while start < end:
        if str[start] != str[end]:
            return False
        start += 1
        end -= 1
    
    return True


# Pattern 3: SLIDING WINDOW
# KEY WORDS: SUBARRAYS/SUBSTRING THAT MATCH CRITERIA
# Example: Find subarrays of size k with max sum
# Time Complexity: O(n) | Space Complexity: O(1)

# Example problem: Subarray of size 'K' with max sum
def max_subarray(arr, k):

    n = len(arr)
    window_sum = sum(arr[:k])
    max_sum = window_sum
    max_sum_start_index = 0

    for i in range(n - k):

        # calculate our current window sum, remove the first num and append the new next num
        window_sum = window_sum - arr[i] + arr[i + k]

        # replace our max sum when we find a better window
        if window_sum > max_sum:
            max_sum = window_sum
            max_sum_start_index += 1
        
    # return our subarray with max sum and our max sum
    return arr[max_sum_start_index:max_sum_start_index+1], max_sum


# Pattern 4: FAST/SLOW POINTER
# KEY WORDS: LINKED LIST/ARRAYS CYCLES OR MIDDLE LINKED LIST
# The fast pointer will essentially catch up to the slow pointer if a cycle is possible
# Time Complexity: O(n) | Space Complexity: O(1)

# Example problem: Detect Linked List cycle
# LinkedList class
class LinkedList:

    def __init__(self, node, next):
        self.node = node
        self.next = next
    
def detect_cycle(head: LinkedList) -> bool:
    
    dummy = head
    slow = fast = dummy

    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next

        if slow is fast:
            return True

    return False


# Pattern 5: LINKED LIST IN PLACE REVERSAL
# KEY WORDS: SWAP NODES/REARRANGE NODES
# Time Complexity: O(n) | Space Complexity: O(1)

# Example problem: Reverse a linked list
# LinkedList class
class LinkedList:

    def __init__(self, node, next):
        self.node = node
        self.next = next

def reverse_list(head: LinkedList):

    # use 3 variables: prev, curr, next
    prev = None
    curr = head

    while curr:
        next = curr.next
        curr.next = prev
        prev = curr
        curr = next

    return prev

# PATTERN 6: MONOTONIC STACK
# KEY WORDS: NEXT SMALLER/GREATER ELEMENT IN ARRAY
# Time Complexity: O(n) | Space Complexity: O(1)

# Example: Next Greater element
# For each index, we want to see how many indexes will take to find the next greater element
# ex: [1, 4, 6, 3, 2, 7] --> yields [1, 1, 3, 2, 1, 0]
def next_greater(arr):
    
    n = len(arr)
    # our stack in this case store the indexes of the number to be evaluated when traversing our list
    stk = []
    res = [0] * n

    # loop our list of numbers
    for i in range(n):
        
        # evaluate our stack items until its empty and the current temp at i is smaller than our top stack item
        while stk and arr[i] > arr[stk[-1]]:
            curr_index = stk.pop()
            res[curr_index] = i - curr_index

        stk.append(i)
    
    return res


# Pattern 7: TOP K ELEMENTS
# KEY WORDS: 'K' LARGEST/SMALLEST/CLOSEST | 'K' MOST FREQUENT
# Using a max/min heap, we can solve these problems | min heap for largest, max heap for smallest 
# The heapq library default to a min heap, so to create a max heap use negative numbers
# Time Complexity: O(nlog k) | Space Complexity: O(n)

# Example problem: return the largest 'k' element | note that we use a min heap here because if we heapify, it would be O(n) + O(n log k)
# ex: [3,2,1,5,6,4] and k = 2 --> yields 5
import heapq

def largest_k_value(arr, k):

    # we build our heap of size k as we go. We keep adding items until we reach a size of k. When full, we add our new element but pop the smallest item, hence the min heap. 
    # We can then return our heap at index k which is at the top of our heap

    min_heap = []

    for num in arr:
        
        # check if we more than k elements, if so pop our min item
        if len(min_heap) < k:
            heapq.heappush(min_heap, num)
        else:
            heapq.heappushpop(min_heap, num)
    
    return min_heap[0]

# Example problem: smallest k value
# ex: [3,2,1,5,6,4] and k = 2 --> yields 2
def smallest_k_value(arr, k):

    # we build our heap of size k as we go. We keep adding items until we reach a size of k. When full, we add our new element but pop the smallest item, hence the min heap. 
    # We can then return our heap at index k which is at the top of our heap

    max_heap = []

    for num in arr:
        
         # check if we more than k elements, if so pop our min item
        if len(max_heap) < k:
            heapq.heappush(max_heap, -num)
        else:
            heapq.heappushpop(max_heap, -num)
    
    return -max_heap[0]

# Example problem: top k frequent elements
# Time complexity: O(n log k)
from collections import Counter
def topKFrequent(nums, k):

    # create our counter of our nums | here our key = number and val = frequency
    counter = Counter(nums)
    min_heap = []

    for key, val in counter.items():

        if len(min_heap) < k:
            heapq.heappush(min_heap, (val, key))
        else:
            heapq.heappushpop(min_heap, (val, key))

    
    return [h[1] for h in min_heap]

# We can also solve this in O(n) since the max frequency possible is the len(arr)
# As such, we can create an array of length of n, representing each frequency. The values that will be stored in each "bucket" are the numbers that have that corresponding frequency
# This is called bucket sort, where we store our items in buckets
def topKFrequent_bucket(nums, k):

    n = len(nums)
    counter = Counter(nums)
    buckets = [0] * (n+1)

    for num, freq in counter.items():

        if buckets[freq] == 0:
            buckets[freq] = [num]
        else:
            buckets[freq].append(num)
    
    res = []

    for i in range(n, -1, -1):
        if buckets[i] != 0:
            # since the answer is unique, we can append the whole bucket. There will not be a case where we have to pick a combination of items in our bucket for our answer
            res.extend(buckets[i])
        if len(res) == k:
            break
    
    return res



# Pattern 8: OVERLAPPING INTERVALS
# Add intervals to our list. Check if previous stored interval overlaps with current, if so modify the previous one. If not add it to our list
# example: merge intervals, meeting times, calendar


# Pattern 9: MODIFIED BINARY SEARCH | when input is not sorted
# KEY WORDS: SEARCHING IN A NEARLY SORTED/ROTATED/UNKNOWN LENGTH/DUPLICATES ARRAY | FINDING PEAK ELEMENT
#            SEARCHING FIRST/LAST OCCURENCE OF ELEMENT | SEARCHING SQUARE ROOT OF NUMBER
# Usually perform binary search with an additional check to see which part of the array is sorted


# Pattern 10: BINARY TREE TRAVERSAL
# PRE/IN/POST ORDER TRAVERSALS | LEVEL ORDERS
# Use InOrder when we have a sorted tree to retrieve values