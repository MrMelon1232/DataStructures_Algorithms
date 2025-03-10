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
# Note: If we find key words like longest substring require extra data structures
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
    return arr[max_sum_start_index:max_sum_start_index+k+1], max_sum


# Pattern 4: FAST/SLOW POINTER
# KEY WORDS: LINKED LIST/ARRAYS CYCLES OR MIDDLE LINKED LIST OR REMOVE NTH NODE
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
# KEY WORDS: NEXT SMALLER/GREATER ELEMENT IN ARRAY | PREV SMALLER/PREV GREATER ELEMENT
# USE A STACK THIS PROBLEM AND A QUEUE FOR SLIDING WINDOW PROBLEMS WHERE WE HAVE TO MAINTAIN ORDER
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
# TIPS/TRICKS:
# 1. Use InOrder when we have a sorted tree to retrieve values (bst)
# 2. Use PreOrder when we want to copy a tree
# 3. Use PostOrer when we want to process the child before parents
# 4. Use level order traversal when we want to explore all nodes at current level


# Pattern 11: DFS/BFS
# KEY WORDS: FINDING PATH BETWEEN 2 NODES | CYCLE | TOPOLOGICAL SORT | # OF CONNECTED COMPONENTS 
# Use a visited set to keep track of seen nodes

# Pattern 12: Matrix traversal patterns
# can use most graph algos 

# Pattern 13: BACKTRACKING
# KEY WORDS: GENERATE COMBINATION OF ALL POSSIBLE SOLUTIONS

# Pattern 14: DYNAMIC PROGRAMMING
# KEY WORDS: BREAKING PROBLEMS INTO SUBPROBLEMS, REPEATING/OVERLAPPING SUBPROBLEMS
#            MAXIMIZING OR MINIMIZE CERTAIN VALUES/COUNT # OF WAYS
#            SUBSTRING/SUBSEQUENCE
# Usual DP arrays store the same data type as the return type of the problem
# Use 2D DP array when we have multiple dimensions/conditions that affect our solution
# Use 1D DP array when we have 1 dimesion/condition affecting our solution.
# Our size of arrays can be n+1 if our base case is 0 where it represents an empty string for examples
# LOOK FOR OVERLAPPING PROBLEMS/SUBPROBLEMS

# DP PATTERNS:
# 1. Fibonacci (2 decisions to make | combination of 2 known solution to create the next one)
# 2. 0/1 Knapsack
# 3. Longest common subsequence | 2D DP ARRAY
# 4. Kadane's algorithm
# 5. Longest Increasing Subsequence | 1D DP ARRAY
# 6. Subset sum
# 7. Matrix Chain Multiplication



# Interesting problems that should be memorized:
'''
2. Longest Palindrome Substring: we want to find the longest palindrome substring
    - a possible solution is to use DP as we want to try multiple combinations of substrings to satisfy our result
    - another solution is to start inwards and expand out using 2 pointers. If the characters are equal, then we have a valid palindrome
    - going with sol 2, we can store our longest substring by comparing our window size. We can get our index by window_size // 2 for the offsets of the indexes
'''
def longestPalindrome(self, s: str) -> str:
        
    # thought process: we can start from the middle of our string and go outwards. If its even length, take the 2 middle characters, if odd, take the middle one. We check for each window if its a palindrome or not. As such, we can explore each index in our string input and do this operation. However this is not as efficient as if we use dynamic programming. When we see the word substring or subsequence, we can attempt a dynamic programming approach. A substring is a palindrome even at the single character length. As such, if we increase the length by 1, we can recheck our condition over and over. If our start is the same as our end, then we have a palindrome if the content inside is also a palindrome. This offers up a tabulation solution that we can use.

    # Potential solution: use a dp array where index i represents the start and index j represents the end of our substring. Our value stored at these indexes are 1 and 0 (true/false) representing if its possible to create a palindrome in that substring. As such, a string is a palindrome if s[start] == s[end] and dp[start+1][end-1] == 1. This works for length bigger than 3. So we need to populate our dp array for lengths 1 and 2

    # n = len(s)
    # dp = [[0] * n for _ in range(n)]

    # # Use variables to keep track of the start and end of our longest palindrome substring
    # start = 0
    # end = 0

    # # check substrings of length 1
    # for i in range(n):
    #     dp[i][i] = 1

    # # check substrings of length 2
    # for i in range(n-1):
        
    #     # check if the 2 letters nearby are palindromes
    #     if s[i] == s[i+1]:
    #         dp[i][i+1] = 1
    #         start = i
    #         end = i+1

    # # check substrings of length 3+ | our first loop creates this difference while our 2nd loop, iterates us through our string
    # for i in range(2, n):
    #     for l in range(n - i):
            
    #         # calculate our last index of our string
    #         r = l + i

    #         # case where we have a matching start and end letter | the inside is also a palindrome
    #         if s[l] == s[r] and dp[l+1][r-1] == 1:
    #             dp[l][r] = 1 

    #             start = l
    #             end = r

    # return s[start:end+1]

    # Time Complexity: O(n^2)
    # Space Complexity: O(n^2)


    # method 2: using in to out searcH. We focus on the centers instead of the bounds, and expand outwards.

    # helper function to process our palindrome expansion
    # returns the length of the current expanded palindrome
    def expand(i, j):
        
        # loop while we're still in bound and we that we have a valid palindrome
        while i >= 0 and j < len(s) and s[i] == s[j]:

            i -= 1
            j += 1
        
        return j - i - 1
    
    # variables to hold our substring bounds
    start = 0
    end = 0
    n = len(s)

    for i in range(n):

        # check the possibility of an odd length palindrome
        odd_length = expand(i, i)    
        # check if we found a new long substring
        if odd_length > end - start + 1:

            # calculate number of items on each side of our palindrome from the center
            dist = odd_length // 2
            start = i - dist
            end = i + dist

        
        # check the possibility of an even length palindrome
        even_length = expand(i, i+1)
        # check if we found a new long substring
        if even_length > end - start + 1:

            # calculate number of items on each side of our palindrome from the center
            dist = even_length // 2
            start = i - dist + 1
            end = i + dist

    return s[start:end+1]

    # Time Complexity: O(n^2) | slightly faster than dp solution since we have less centers than bounds
    # Space Complexity: O(1)
    
'''
3. Word Break: we want to find if we can make our string s with the word bank we are provided
    - we use a 1D DP array to keep track of if its possible to make our word from s[0:i]
    - the goal is to have our end string able to be made so dp[n] should be true 
    - why do we use 1D DP ARRAY: we want to find if the entire string is valid. This makes it that our solution depends on 1 dimension
'''

def wordBreak(self, s: str, wordDict: List[str]) -> bool:
        
    # thought process: we can use a sliding window approach where we open our window size until we find our first valid word. If found, we slide our left pointer.

    # problem encountered, if we need to make a combination of the word but the way we're checking with solution 1 does not account for that. For example, if our word is "aaaaaaaa" and we have wordDict ["aaaa", "aaa"] then the issue here is that we will only check the lowest length string. As such, we need to explore multiple combinations that will arrive us to our solution. An example of such is to use dynamic programming. Where we explore all valid paths at our current index. We can still use 2 pointers to keep track of the boundaries of our string, hence storing our result.

    # Potential solution: we can use a 1D DP array because we need to check if the entire string is valid. This means we dont check combinations/multiple substrings, hence creating only 1 condition/dimension of the problem, where the ith index depends on the indexes before it. We dont have 2 values that modify our solution. 

    n = len(s)

    # use a set to store our words for faster lookup
    knownWords = set(wordDict)

    # create our 1D dp array | each index represents if its possible to make that word from our dictionary 
    # each index i will depend on it previous answers. If s[j:i] is valid, then s[j] must also be valid as in, its prefix string must be possible to create
    dp = [False] * (n+1)

    # base case: basically the empty string can be segmented
    dp[0] = True

    # loop for n+1 because 0 is our base case
    for i in range(1, n+1):
        for j in range(i):

            # check if our substring is a word in our wordDict | we also check if dp[j] is valid, meaning that substring 0-j can be made from our knownWords
            if s[j:i] in knownWords and dp[j] == True:
                dp[i] = True

    return dp[n]

    # Time Complexity: O(n^2)
    # Space Complexity: O(n)

'''
4. 3 sum: we want to find all the numbers that can sum up to 0 and avoid duplicates
    - sort our list of numbers
    - pick a number as our placeholder and perform 2 sum on the other numbers
    - we skip numbers that are identical to its previous when we loop for potential 2 sums with our placeholder number
'''

from collections import defaultdict
class Solution:
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        
        # thought process: we need the total combination of triplets that can make our total 0. We can take an initial element and mark it as our target. As such, we now need 2 more elements that can sum up to our target, similar to 2 sum problem. 
        # nums.sort()
        # res = set()
        # # A hash map to store our values
        # d = {}

        # # fill our dictionary
        # for i in range(len(nums)):
        #     d[nums[i]] = i

        # # loop through our list of nums
        # # for each number in our list, it is going to act as a placeholder
        # for i in range(len(nums)):
        #     target = nums[i]

        #     # loop through the rest of our numbers
        #     for j in range(i+1, len(nums)):

        #         diff = target + nums[j]

        #         if -diff in d and d[-diff] > j:
        #             res.add((target, nums[j], -diff))
        
        # return [list(triplets) for triplets in res]


        # Method 2: using 2 pointer
        # we can sort our list of numbers and using the same logic as method 1 with a hashmap, we instead search for the missing target by squeezing our array, hence speeding up the process. To avoid any duplicates, when we find another matching number as the previous one when traversing our list, we skip it
        
        nums.sort()
        res = []
        n = len(nums)

        # loop though all our numbers | each number at index i is our placeholder for performing 2 sum
        for i in range(n):
            
            # exit when we find a positive number | since our nums are sorted, we wont find another combination for 0
            if nums[i] > 0:
                break

            # skip number if we have already seen it | prevents duplicates
            if i > 0 and nums[i] == nums[i-1]:
                continue
            
            # define our 2 pointers
            start = i + 1
            end = n-1

            # loop while we have a valid range | note that to reach our target of 0, we can move either our left/right pointer depending on our condition
            while start < end:
                
                target_sum = nums[i] + nums[start] + nums[end]

                # append our numbers if we found a combination
                if target_sum == 0:
                    res.append([nums[i], nums[start], nums[end]])
                    start += 1
                    end -= 1

                    # skip the number if its the same
                    while start < end and nums[start] == nums[start-1]:
                        start += 1
                    
                    # skip the number if its the same
                    while start < end and nums[end] == nums[end-1]:
                        end -= 1

                # if our target sum is smaller than 0, we need a bigger number
                elif target_sum < 0:
                    start += 1
                    
                # if our target sum is bigger than 0, we need a smaller number
                else:
                    end -= 1
                   
            
        return res

        # Time Complexity: O(n^2)
        # Space Complexity: O(1)


'''
5. Longest Increasing Subsequence: we want to find the longest increasing subsequence
    - same thing with word break, we can use a 1D DP array that keeps track
    - our 1D DP array represents the longest increasing subsequence from s[0:i] for each index i
    - our solution will hence be at the max of our dp array
'''
def lengthOfLIS(self, nums: List[int]) -> int:
    
    # thought process: we want the longest length of increasing subsequence in our array. The first idea we can think of is itearing through our list of nums and having a placeholder for our smallest element at index i. Then for each index, we iterate from our placeholder index i till the end of our array while keeping track of a new found minimum and length of increasing sequence count. When we find an element bigger than the our curr_min, we increment our current longest sequence var and replace the curr_min with our new number. If we encounter a smaller number than our curr_min but bigger than our placeholder number, we reset our curr longest count and go from there. This will result in a O(n^2) time complexity solution which is not ideal. 

    # Since we want to compare multiple combination of numbers that form an increasing subsequence and find the max number of them, we can use backtracking to do so. We can view our list of nums as each index representing the longest increasing subsequence that ends at that index. In our dp array, our current index represents the max increasing subsequence till our number at our current index

    n = len(nums)
    dp = [1] * n

    for i in range(1, n):
        for j in range(i):
            
            # increment our longest subsequence at i if our number at j is smaller than our number at i
            if nums[j] < nums[i]:
                
                # keep the max of longest subsequence or store the new one
                # we add 1 here for the current number at index i that counts towards our subsequence
                dp[i] = max(dp[i], dp[j] + 1)


    return max(dp)

    # Time Complexity: O(n^2)
    # Space Complexity: O(n)

'''
6. Longest Common Subequence: we want the find the longest common subsequence between 2 strings
    - use a 2D DP array because we have 2 dimensions to keep track of: the index of s1 and index of s2
    - our 2D DP array represents the longest common subsequence between of s1[0:i] and s2[0:j]
    - 2 conditions to check every time we loop:
        1. matching characters = skip index of both strings
        2. missmatch characters = explore diff paths(skip index i or skip index j) | take the max between both
'''
def longestCommonSubsequence(self, text1: str, text2: str) -> int:
    
    # thought process: the brute force solution is to check each index of string 1 with all characters of string 2. Pursue paths where we find a match and if not we continue by either skipping the curr char at text1 or text2. This will result in a O(2^(m*n)) time complexity, since at each decision revolves either adding our current character or not. 
    
    # method 1: using memoization
    # here, our memoization stores the longest common subsequence until our index i and j of our 2 strings
    # m = len(text1)
    # n = len(text2)
    # memo = [[-1] * (n + 1) for _ in range(m + 1)]
    # def backtrack(i, j):
        
    #     # check if the answer is in our memo
    #     if memo[i][j] != -1:
    #         return memo[i][j]

    #     # base case: if one of our strings has been fully processed, then return 0
    #     if i == len(text1) or j == len(text2):
    #         memo[i][j] = 0
    #         return memo[i][j]

    #     # case where we have a matching character
    #     if text1[i] == text2[j]:
    #         memo[i][j] = 1 + backtrack(i + 1, j + 1)
    #         return memo[i][j]
        
    #     # case where we make our choice. We get the max of either choices, where choice 1 is continuing our str1 and choice 2 is continuing our str2
    #     else:
    #         memo[i][j] = max(backtrack(i + 1, j), backtrack(i, j + 1))
    #         return memo[i][j]
    
    # return backtrack(0, 0)


    # Method 2: tabulation
    # use a bottom up approach by storing the LCS in our dp array, using our index i of string 1 and index j of string 2
    # Since this is a 2D DP problem, visualize a table where we have index i and j as our position in our dp table
    # We want to optimize the problem in a way that we arrive to our optimal solution which is located at the bottom right and down of our table. Same way how we want to build our sub problems solution in a 1D DP problem to arrive at the end of our dp array.


    # DP 2D ARRAY thought process: we have 2 conditions to process at each step.
    # 1. If we have a matching character, we move on diagonally in our matrix, since this represents a match and we want to now skip the current char of both text1 and text2, hence increasing i and j by 1. 
    # 2. If we don't get a match, then we want to explore the paths where we have different characters, these are either at i - 1 or j - 1. We take their max between both

    m = len(text1)
    n = len(text2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    # we loop through our DP Grid/table | start at (1,1) because our first row and column are all 0s
    for i in range(1, m+1):
        for j in range(1, n+1):

            # case where we have a match of characters | note that dp[i-1][j-1] is the max at the previous comparison
            if text1[i-1] == text2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            
            # have to take the max between our 2 explored paths
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    
    return dp[m][n]

    # Time complexity: O(m*n)
    # Space complexity: O(m*n)



'''
23. Merge K sorted Lists: we want to merge k sorted lists onto a new linked list
    - we can use a min heap to store our nodes val, index in our input and our node itself
    - loop in our min heap and pop the min item, add it to our linked list and push the next reference of the popped node
'''
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
import heapq

def mergeKLists(self, lists: List[Optional[ListNode]]) -> Optional[ListNode]:

    # attempt 2: instead of using heapify, we push the node references onto our min heap and keep popping and adding items into it

    # store in our min_heap the node reference and its value | if we have equal value, we can use its index to get the first one
    min_heap = []

    # Push all nodes onto our min heap
    for i, node in enumerate(lists):
        if node:
            heapq.heappush(min_heap, (node.val, i, node))

    dummy = ListNode()
    curr = dummy

    # loop until we popped all our items from our min_heap
    while min_heap:
        
        val, index, node = heapq.heappop(min_heap)

        # Add the next node reference onto our min heap if it exists
        if node.next:
            heapq.heappush(min_heap, (node.next.val, index, node.next))


        # Add our node to our new linked list
        curr.next = node
        curr = curr.next

    return dummy.next


'''
417. Pacific Atlantic Water Flows
There is an m x n rectangular island that borders both the Pacific Ocean and Atlantic Ocean. The Pacific Ocean touches the island's left and top edges, and the Atlantic Ocean touches the island's right and bottom edges.

The island is partitioned into a grid of square cells. You are given an m x n integer matrix heights where heights[r][c] represents the height above sea level of the cell at coordinate (r, c).

The island receives a lot of rain, and the rain water can flow to neighboring cells directly north, south, east, and west if the neighboring cell's height is less than or equal to the current cell's height. Water can flow from any cell adjacent to an ocean into the ocean.

Return a 2D list of grid coordinates result where result[i] = [ri, ci] denotes that rain water can flow from cell (ri, ci) to both the Pacific and Atlantic oceans.
'''

from collections import deque

def pacificAtlantic(self, heights: List[List[int]]) -> List[List[int]]:
    
    # thought process: similar to the other island problem, we can perform a DFS on each grid coordinate and append the current coordinates if water can flow to both pacific and atlantic oceans. Performing DFS will help us navigate each possible route. There are 2 conditions to initiate a stopping point:
    # 1. We reached an adjacent position in the grid that water can not flow to
    # 2. We reached either the pacific ocean or the atlantic ocean
    # Note: we must reach both oceans in order to append our coordinate
    # Time complexity: O(2 * m x n) --> O(m x n)

    # approach 2: put all edge locations into our queues for atlantic and pacific and then start our bfs traversal by going inwards. 

    # queue/set for pacific
    p_que = deque()
    p_seen = set()
    
    # queue/set for atlantic
    a_que = deque()
    a_seen = set()
    
    m, n = len(heights), len(heights[0])

    # add edge grid locations to our queue that are near our pacific ocean
    for j in range(n):
        p_que.append((0, j))
        p_seen.add((0, j))
        
    for i in range(1, m):
        p_que.append((i, 0))
        p_seen.add((i, 0))

    # add edge grid locations to our queue that are near our atlantic ocean
    for i in range(m):
        a_que.append((i, n - 1))
        a_seen.add((i, n - 1))
        
    for j in range(n - 1):
        a_que.append((m - 1, j))
        a_seen.add((m - 1, j))

    # bfs function to check adjacent locations for both oceans | only visited nodes are added to our set, so we take the intersection of both (common results from both sets)
    def get_coords(que, seen):
        while que:
            i, j = que.popleft()
            for i_off, j_off in [(0, 1), (1, 0), (-1, 0), (0, -1)]:
                r, c = i + i_off, j + j_off
                if 0 <= r < m and 0 <= c < n and heights[r][c] >= heights[i][j] and (r, c) not in seen:
                    seen.add((r, c))
                    que.append((r, c))
        
    get_coords(p_que, p_seen)
    get_coords(a_que, a_seen)
    return list(p_seen.intersection(a_seen))

# Time Complexity: O(m*n)
# Space Complexity: O(m*n)


'''
424. Longest Repeating Character Replacement
You are given a string s and an integer k. You can choose any character of the string and change it to any other uppercase English character. You can perform this operation at most k times.

Return the length of the longest substring containing the same letter you can get after performing the above operations.

Example 1:

    Input: s = "ABAB", k = 2
    Output: 4
    Explanation: Replace the two 'A's with two 'B's or vice versa.
'''

def characterReplacement(self, s: str, k: int) -> int:
        
    # thought process: we use a sliding window approach. For each window, we check the current number of characters that can be swapped. That number has to be less or equal to k. 
    # We know this by keeping track of the count of our letters in our window size. window_size (r-l+1) - most dominant letter count = total letters we can swap. If it passes k, we need to slide our window.

    freq_count = [0] * 26
    n = len(s)
    longest_repeating_count = 0               
    l = 0

    # loop through our list of characters
    for r in range(n):

        # increase our frequency count for the current letter
        freq_count[ord(s[r]) - ord('A')] += 1

        # calculate how many k operations are allowed in our current window size
        # note that its done in the while loop so its updated at each iteration
        
        # move our left pointer until we have a valid window
        while ((r-l+1) - max(freq_count)) > k:
            # decrement our freq for the char at the left pointer
            freq_count[ord(s[l]) - ord('A')] -= 1 
            l += 1
        
        longest_repeating_count = max(longest_repeating_count, (r-l+1))

    return longest_repeating_count


    '''
    Alternate code by using a dictionary:

    class Solution:
    def characterReplacement(self, s: str, k: int) -> int:
        l = 0
        count = {}
        maxC = 0
        result = 0
        for r in range(len(s)):
            count[s[r]] = count.get(s[r], 0) + 1
            maxC = max(maxC, count[s[r]]) 
            while (r - l + 1) - maxC > k:
                count[s[l]] -= 1
                l += 1
            result = max(result, r - l + 1)
        return result
    '''

'''
567. Permutation in String
Given two strings s1 and s2, return true if s2 contains a permutation of s1, or false otherwise.

In other words, return true if one of s1's permutations is the substring of s2.

Example 1:

    Input: s1 = "ab", s2 = "eidbaooo"
    Output: true
    Explanation: s2 contains one permutation of s1 ("ba").
'''
from collections import Counter
def checkInclusion(self, s1: str, s2: str) -> bool:
    
    # thought process: we want to know if a permutation of s1 is contained within s2. Now we can use a sliding window approach of size len(s1) and check if the current window is a permutation of s1. We can use a Counter to store the frequency of each letter in our string s1 and see if the counter for our current window matches it. If so return true. We can also create our counter for s2 and modify it while going in our loop, which saves time and space.
    # Note: we can also solve this with a count freq of alphabet numbers using their ascii as their index

    # Time Complexity: O(n) | Space Complexity: O(n)

    l = 0
    n = len(s2)
    s1_freq = Counter(s1)
    s2_freq = Counter(s2[:len(s1)])

    if s1_freq == s2_freq:
        return True

    for r in range(len(s1), n):
        
        s2_freq[s2[l]] -= 1

        if s2_freq[s2[l]] == 0:
            del s2_freq[s2[l]]
        s2_freq[s2[r]] += 1

        l += 1

        if s1_freq == s2_freq:
            return True
    
    return False


'''
295. Find Median from Data Stream
The median is the middle value in an ordered integer list. If the size of the list is even, there is no middle value, and the median is the mean of the two middle values.

For example, for arr = [2,3,4], the median is 3.
For example, for arr = [2,3], the median is (2 + 3) / 2 = 2.5.
Implement the MedianFinder class:

MedianFinder() initializes the MedianFinder object.
void addNum(int num) adds the integer num from the data stream to the data structure.
double findMedian() returns the median of all elements so far. Answers within 10-5 of the actual answer will be accepted.
 

Example 1:

    Input
    ["MedianFinder", "addNum", "addNum", "findMedian", "addNum", "findMedian"]
    [[], [1], [2], [], [3], []]
    Output
    [null, null, null, 1.5, null, 2.0]

    Explanation
    MedianFinder medianFinder = new MedianFinder();
    medianFinder.addNum(1);    // arr = [1]
    medianFinder.addNum(2);    // arr = [1, 2]
    medianFinder.findMedian(); // return 1.5 (i.e., (1 + 2) / 2)
    medianFinder.addNum(3);    // arr[1, 2, 3]
    medianFinder.findMedian(); // return 2.0
'''
import heapq
class MedianFinder:

    def __init__(self):
        self.left_max_heap = []
        self.right_min_heap = []

    def addNum(self, num: int) -> None:

        # store num in our left heap if its within its range 
        if not self.left_max_heap or num <= -self.left_max_heap[0]:
            heapq.heappush(self.left_max_heap, -num)
        else:
            heapq.heappush(self.right_min_heap, num)

        # balance both of our heaps | we want ideally equal length or left having length of right + 1
        if len(self.left_max_heap) > len(self.right_min_heap) + 1:
            heapq.heappush(self.right_min_heap, -heapq.heappop(self.left_max_heap))
        
        # case where our right heap is bigger
        elif len(self.right_min_heap) > len(self.left_max_heap):
            heapq.heappush(self.left_max_heap, -heapq.heappop(self.right_min_heap))

    def findMedian(self) -> float:

        # case where our length is odd
        if len(self.left_max_heap) > len(self.right_min_heap):
            return float(-self.left_max_heap[0])

        # case where our length is even
        else:
            return float((-self.left_max_heap[0] + self.right_min_heap[0]) / 2)


# thought process: since we want to sort our data, adding numbers will cause it to be unsorted. 
# The goal here is to use 2 heaps, 1 max heap for the left half of our array and a min heap for the right part of our array. 
# As such, the top element of the map heap and the top element of the min heap on the left will be our 2 centered items. 
# Make sure that the len(left_heap) <= len(right_heap) + 1
# All of our operations are O(log(n))


# Your MedianFinder object will be instantiated and called as such:
# obj = MedianFinder()
# obj.addNum(num)
# param_2 = obj.findMedian()

'''
Detect cycle in undirected graph
    - we store the parent of each node so we dont mistake visited nodes with our parents
'''

from collections import deque, defaultdict

def has_cycle_undirected_bfs(graph):
    visited = set()
    
    # loop here in case of disconnected graphs. If all connected, this loop is not necessary
    for start in graph:  # Check all components
        if start in visited:
            continue

        queue = deque([(start, -1)])  # Queue stores (node, parent)
        while queue:
            node, parent = queue.popleft()
            
            if node in visited:
                continue
            
            visited.add(node)
            for neighbor in graph[node]:
                if neighbor not in visited:
                    queue.append((neighbor, node))
                elif neighbor != parent:  # Cycle detected
                    return True
    return False

# Example Usage
graph_undirected = defaultdict(list)
edges = [(0,1), (1,2), (2,3), (3,0)]  # Cycle exists
for u, v in edges:
    graph_undirected[u].append(v)
    graph_undirected[v].append(u)

print(has_cycle_undirected_bfs(graph_undirected))  # Output: True



'''
416. Partition Equal Subset Sum | 0/1 KNAPSACK PROBLEM
Given an integer array nums, return true if you can partition the array into two subsets such that the sum of the elements in both subsets is equal or false otherwise.

Example 1:

    Input: nums = [1,5,11,5]
    Output: true
    Explanation: The array can be partitioned as [1, 5, 5] and [11].

Example 2:

    Input: nums = [1,2,3,5]
    Output: false
    Explanation: The array cannot be partitioned into equal sum subsets.
'''

# This is a 0/1 Knapsack problem similar to coin change. Our dp array represents the sum possible to be reached using the numbers in our array
def canPartition(self, nums: List[int]) -> bool:
    
    # thought process: we want a single valid combination of numbers from our list of nums to split it into 2 subsets that have equal sum. The sum of both arrays have to equal half of the current total sum. Our goal is to find if its possible to reach our target sum with a combination of some numbers. Hence, this hints at a dynamic programming question where we have to look at different combinations. Each index in our dp array can represent if its possible to reach our target_sum. We're going to have a DP array of size taget + 1 | we want to return dp[target] as our answer. This is a 0/1 knapsack problem similar to the coin change problem. 

    total_sum = sum(nums)

    # we can return False if we cant divive our sum by 2
    if total_sum % 2 != 0:
        return False
    
    
    target = total_sum // 2
    n = len(nums)
    dp = [False] * (target + 1)
    # base case: because reaching sum 0 is possible
    dp[0] = True

    for num in nums:  # Process each number first (important for 0/1 Knapsack)
        for i in range(target, num - 1, -1):  # Iterate backwards to avoid reusing elements
            
            # From our target, is it possible to find a sum that reaches i using our num
            # we check dp[i - num] because we want to check if a number can be summed up to that one, if so then that number + our curren num = target
            dp[i] = dp[i] or dp[i - num]
    

    return dp[target]


    # Time Complexity: O(n*k) where n = length of nums and k = our target
    # Space Complexity: O(k) where k = target


'''
322. Coin Change | UNBOUNDED KNAPSACK PROBLEM
You are given an integer array coins representing coins of different denominations and an integer amount representing a total amount of money.

Return the fewest number of coins that you need to make up that amount. If that amount of money cannot be made up by any combination of the coins, return -1.

You may assume that you have an infinite number of each kind of coin.

'''
def coinChange(self, coins: List[int], amount: int) -> int:

    # Attempt 2: use a 1D DP array where each index represents if that total amount can be reached by using the min of coins

    # initialize our dp array of inf numbers
    dp = [float('inf')] * (amount+1)

    # we know that to reach sum 0, we just need 0 coins
    dp[0] = 0

    # for each coin, evaluate if the difference can be reached as each index represents the min amount of coins to reach that amount (remains float('inf') if not possible)
    for coin in coins:
        for i in range(coin, amount+1):
            dp[i] = min(dp[i], dp[i-coin] + 1) # dp[i-coin] + 1 represents us solving at i-coin with our current coin being counted (by adding +1)

    
    return dp[amount] if dp[amount] != float('inf') else -1
    

    # Time Complexity: O(n * amount)
    # Space Complexity: O(amount)
    

'''
IMPORTANT FOR DP SOLUTIONS OF KNAPSACK TYPE
Forward iteration (range(num, target + 1)) → allows repetitions (like Coin Change).
Backward iteration (range(target, num - 1, -1)) → ensures each number is used at most once (like 0/1 Knapsack or Partition Equal Subset Sum).
'''


'''
208. Implement Trie (Prefix Tree)
A trie (pronounced as "try") or prefix tree is a tree data structure used to efficiently store and retrieve keys in a dataset of strings. There are various applications of this data structure, such as autocomplete and spellchecker.

Implement the Trie class:

Trie() Initializes the trie object.
void insert(String word) Inserts the string word into the trie.
boolean search(String word) Returns true if the string word is in the trie (i.e., was inserted before), and false otherwise.
boolean startsWith(String prefix) Returns true if there is a previously inserted string word that has the prefix prefix, and false otherwise.
'''

class Trie:

    def __init__(self):
        self.trie = {}

    def insert(self, word: str) -> None:
        d = self.trie

        # insert each character as a key in our hashmap but make each key point to another dictionary
        for char in word:

            # if the current character is not in our dictionary, then add it (indicates that its the start of a word with a new letter
            if char not in d:
                d[char] = {}
            # move on to our sub dictionary
            d = d[char]

        # This marks the end of our string, in our most nested dictionary. If we find '.', it means that string is a word
        d['.'] = '.'


    def search(self, word: str) -> bool:
        d = self.trie

        for char in word:
            if char not in d:
                return False
            d = d[char]
    
        return '.' in d

    def startsWith(self, prefix: str) -> bool:
        d = self.trie

        for char in prefix:
            if char not in d: 
                return False
            d = d[char]

        return True
        

# thought process: the problem/challenge here is to figure out how to detect prefixes here in our storage of words. The trick here is to use a tree structure to store all of our prefixes/combination of strings that form our prefixes.

# the other issue is to store multiple words, so we can't just use a single tree node. The trick is to use a dictionary, where each letter is a key and has a dictionary of child nodes/letters

# the trick here is store dictionaries within dictionaries. Our keys are letters and its values are a sub dictionary of the rest of its letters. Hence, we can make multiple words, similar to a tree. 
# This is the reason why its called a trie data structure | also named prefix tree | used as a key value search for optimized searches of words/prefixes and deletions and insertions


# Your Trie object will be instantiated and called as such:
# obj = Trie()
# obj.insert(word)
# param_2 = obj.search(word)
# param_3 = obj.startsWith(prefix)

'''
402. Remove K Digits
Given string num representing a non-negative integer num, and an integer k, return the smallest possible integer after removing k digits from num.
    - THIS IS A MONOTONOMIC STACK PROBLEM

'''
class Solution:
    def removeKdigits(self, num: str, k: int) -> str:
        
        # thought process: after evaluating the problem and the conditions we are searching for, we can deduce that we need to use a monotonomic stack. 
        # We want to essentially prioritize removing bigger elements than smaller ones. Similarly, we want to also prioritize bigger numbers on the left than the right. 
        # Hence use a stack to keep track of our numbers and pop those that are deemed to big. This will result in us wanting an increasing sequence of numbers. So overall 3 scenarios:
        # 1. If we have strictly increasing numbers, remove k numbers at the end
        # 2. If the current number is smaller to our top stack number, we pop our stack (remove the bigger element at the left)
        # 3. If we have 'k' numbers left to use then we remove k numbers at the end

        # this results in us looking for the next greater element, a pattern common for monotonomic stacks

        stk = []

        for c in num:
            
            # check if the current number is smaller to the top stack item
            while stk and k > 0 and stk[-1] > c:
                stk.pop()
                k -= 1
        
            stk.append(c)

        # store our list of letters in our result
        res = ''.join(stk[:len(stk) - k])
      
        # remove leading 0s
        res = res.lstrip("0")

        return res if res != "" else "0"

'''
79. Word Search
Given an m x n grid of characters board and a string word, return true if word exists in the grid.

The word can be constructed from letters of sequentially adjacent cells, where adjacent cells are horizontally or vertically neighboring. The same letter cell may not be used more than once.

    - This is a graph problem that also involves backtracking. Similar to number of islands or max area of islands, instead of using iterative dfs, we use a backtracking dfs function because we need to undo our marked node after going through it
    - Using the same logic, we perform dfs on each square and explore neighbors but we undo our marked node so we can use it for another word.
'''
def exist(self, board: List[List[str]], word: str) -> bool:
    
    # thought process: since we dont have a recurring sub problem or repeating smaller solutions, we can't use dynamic programming to store our results. As such, For each case, we have to explore its neighbors to find if we have a valid string. While exploring our options, if we find the next letter, we can explore that option. If we come to a dead end, we backtrack to our previous letter and explore other paths. 
    
    m = len(board)
    n = len(board[0])

    def dfs(i, j, index):
        # Base case: if we've matched the entire word, return True
        if index == len(word):
            return True

        # Base case: if out of bounds or the letter doesn't match
        if i < 0 or i >= m or j < 0 or j >= n or board[i][j] != word[index]:
            return False

        # Mark the current cell as visited
        temp = board[i][j]
        board[i][j] = '#'

        # Explore all 4 adjacent cells
        for i_off, j_off in [(1, 0), (0, 1), (-1, 0), (0, -1)]:
            i_index = i + i_off
            j_index = j + j_off
            if dfs(i_index, j_index, index + 1):
                return True

        # Backtrack: unmark the current cell
        board[i][j] = temp
        return False

    # Loop through the board to find the first letter of the word
    for i in range(m):
        for j in range(n):
            if board[i][j] == word[0]:  # Found the first letter
                if dfs(i, j, 0):  # Start backtracking
                    return True
    
    return False

    # Time Complexity: O((m*n)**2)
    # Space Complexity: O(L)


'''
39. Combination Sum
Given an array of distinct integers candidates and a target integer target, return a list of all unique combinations of candidates where the chosen numbers sum to target. You may return the combinations in any order.

The same number may be chosen from candidates an unlimited number of times. Two combinations are unique if the frequency of at least one of the chosen numbers is different.

The test cases are generated such that the number of unique combinations that sum up to target is less than 150 combinations for the given input.

    - this is a classic backtracking problem, they mostly have this sort of format
    - reminder: use a for loop in backtracking when we have multiple choices and not just pick or dont pick number

Example 1:

    Input: candidates = [2,3,6,7], target = 7
    Output: [[2,2,3],[7]]
    Explanation:
    2 and 3 are candidates, and 2 + 2 + 3 = 7. Note that 2 can be used multiple times.
    7 is a candidate, and 7 = 7.
    These are the only two combinations.
Example 2:

    Input: candidates = [2,3,5], target = 8
    Output: [[2,2,2,2],[2,3,3],[3,5]]
'''
def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
    
    # initial thoughts: once again, we want all unique combinations of a solution for a problem. This hints us that a backtracking approach is suggested. In this case, we only append a solution when the target that we will be passing onto to our backtrack function is 0. The same case that we only append a number to solution if that number does not make the target reach negative numbers

    # second attempt: we find that our initial attempt is correct but we have to be able to distinguish 2 similar solutions with the same frequency of candidatess in the solution. Potential solution: we could make a frequency counter for each solution and verify if it already exists

    # third attempt: we can use an index tracker to track the current index and make sure we only append and backtrack on items after our current index to avoid duplicates

    res = []
    sol = []
    
    def backtrack(target, start):

        # base case: if we reached our target, add it to our answer
        if target == 0:
            res.append(sol[:])
            return
        
        for i in range(start, len(candidates)):  # Ensure only forward exploration
            if target >= candidates[i]:  # Only proceed if it's a valid choice
                sol.append(candidates[i])
                backtrack(target - candidates[i], i)  # Allow reuse of same element | else it would be i+1
                sol.pop()
    
    backtrack(target, 0)  # Start from index 0
    return res


'''
236. Lowest Common Ancestor of a Binary Tree
Given a binary tree, find the lowest common ancestor (LCA) of two given nodes in the tree.

According to the definition of LCA on Wikipedia: “The lowest common ancestor is defined between two nodes p and q as the lowest node in T that has both p and q as descendants (where we allow a node to be a descendant of itself).”
'''

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
    
    # thought process: we want to find the LCA of the current binary tree. Use a DFS. For every root node, we want to find if its possible to reach node p and node q. We then return the root of that traversal.
    # There are 3 possible cases:
    # 1. p and q are in different subtrees
    # 2. q is an ancestor of p
    # 3. p is an ancestor of q

    # edge case:
    if not root:
        return None

    # case where we find out node
    if root == p or root == q:
        return root

    # Explore our subtrees | returns our node if we find a match, if not returns its 
    left = self.lowestCommonAncestor(root.left, p, q)
    right = self.lowestCommonAncestor(root.right, p, q)

    # if we found our target nodes in both of our subtrees
    if left and right:
        return root
    else:
        # return the next level of nodes
        return left or right


    # Time Complexity: O(n)
    # Space Complexity: O(1)


'''
310. Minimum Height Trees
A tree is an undirected graph in which any two vertices are connected by exactly one path. In other words, any connected graph without simple cycles is a tree.

Given a tree of n nodes labelled from 0 to n - 1, and an array of n - 1 edges where edges[i] = [ai, bi] indicates that there is an undirected edge between the two nodes ai and bi in the tree, you can choose any node of the tree as the root. When you select a node x as the root, the result tree has height h. Among all possible rooted trees, those with minimum height (i.e. min(h))  are called minimum height trees (MHTs).

Return a list of all MHTs' root labels. You can return the answer in any order.

The height of a rooted tree is the number of edges on the longest downward path between the root and a leaf.

 
Example 1:

    Input: n = 4, edges = [[1,0],[1,2],[1,3]]
    Output: [1]
    Explanation: As shown, the height of the tree is 1 when the root is the node with label 1 which is the only MHT.
    Example 2:
'''
from collections import defaultdict
from collections import deque
def findMinHeightTrees(self, n: int, edges: List[List[int]]) -> List[int]:
    
    # thought process: we're given a graph and we want to find which root node can be used to create a MHT. There are 2 factors to determine. 1. Find the min height of our tree | 2. Find possible root that have this height

    # brute force solution is to determine the height of every tree and return the min height.
    # solution 1: start from the leaves of the tree/graph. These can't be the root of the tree because they have the least edges/neighbors. Remaining nodes will be our result. There can only be at most 2 root nodes. Travel to the middle our list of nodes (so length / 2)
    # This is similar to topological sort, as we want to process the leaves first and go inwards, hence creating a sort of dependency

    # edge case
    if n == 1:
        return [0]

    # create a adjacency list
    adj = defaultdict(list)
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)


    # count the edges of our nodes (similar to topological sort)
    leaves = deque()
    indegree = [0] * n
    for src, neighbors in adj.items():
        # if they have 1 neighbor, they must be leaves
        if len(neighbors) == 1:
            leaves.append(src)
        indegree[src] = len(neighbors)
    
    while leaves:
        
        # return the remaining nodes | we can have at most 2
        if n <= 2:
            return list(leaves)

        # loop per order level
        for i in range(len(leaves)):

            node = leaves.popleft()
            n -= 1

            # check neighbors
            for nei in adj[node]:
                indegree[nei] -= 1
                
                # check if this node became a leaf
                if indegree[nei] == 1:
                    leaves.append(nei)

        
    # Time Complexity: O(n)
    # Space Complexity: O(1)


'''
542. 01 Matrix
Given an m x n binary matrix mat, return the distance of the nearest 0 for each cell.

    - BFS BECAUSE OF SHORTEST PATH | DP ALSO OK

The distance between two cells sharing a common edge is 1.
'''
from collections import deque
def updateMatrix(self, mat: List[List[int]]) -> List[List[int]]:
    
    # thought process: we want the distance of the nearest 0 for each cell. As such, we can just modify our input matrix. Each location in our matrix will represent the distance to the nearest 0. We can think of this almost as a DP problem, where we have overlappnig subproblems. Since our subproblems rely on previous answers, we need to do 2 passes:
    # 1 pass from top to bottom left to right | 1 pass in reverse order
    # Doing only 1 pass will not work because our grid relies on all adjacent locations and we did not process the right and bottom subproblems yet. 

    # m = len(mat)
    # n = len(mat[0])

    # # first pass: top->bot | left->right
    # for i in range(m):
    #     for j in range(n):
            
    #         # case where we find a 1
    #         if mat[i][j] > 0:
                
    #             # placeholder max value for our neighbors
    #             top = float('inf')
    #             left = float('inf')

    #             # check its adjacent neighbors from top and left | we want the min path to a 0
    #             if i > 0:
    #                 top = mat[i-1][j]
                
    #             if j > 0:
    #                 left = mat[i][j-1]
            
    #             # update the cell with the minimum value found from neighbors + 1
    #             mat[i][j] = min(top, left) + 1


    # # second pass: bot->top | right->left
    # for i in range(m-1, -1, -1):
    #     for j in range(n-1, -1, -1):

    #         # case where we find a non 0 number
    #         if mat[i][j] > 0:
                
    #             # placeholder for our right and bottom
    #             right = float('inf')
    #             bottom = float('inf')

    #             # check our adjacent grid locations to the right and to the left
    #             if i < m-1:
    #                 right = mat[i+1][j]
                
    #             if j < n-1:
    #                 bottom = mat[i][j+1]

    #             # update the cell with the minimum value found from neighbors + 1
    #             mat[i][j] = min(mat[i][j], right + 1, bottom + 1)
    
    # return mat
                

    # Time Complexity: O(m * n)
    # Space Complexity: O(1)
    
    # bfs method
    q = deque()
    m = len(mat)
    n = len(mat[0])

    # store all of our 0s for each cell
    for i in range(m):
        for j in range(n):

            if mat[i][j] == 0:
                q.append((i, j))
            
            else:
                # if its not a 0, mark the node as unvisited
                mat[i][j] = -1 

    

    # process our 0s and process its neighbors
    while q:

        i, j = q.popleft()

        for i_off, j_off in [(0,1), (1, 0), (-1, 0), (0, -1)]:
            
            i_index = i + i_off
            j_index = j + j_off

            # make sure we are within bounds and it has not yet been visited
            if 0 <= i_index < m and 0 <= j_index < n and mat[i_index][j_index] == -1:
                mat[i_index][j_index] = mat[i][j] + 1
                q.append((i_index, j_index))
    
    return mat
            

'''
253. Meeting Rooms II
Description
Given an array of meeting time intervals intervals where intervals[i] = [starti, endi], return the minimum number of conference rooms required.

Example 1:

    Input: intervals = [[0,30],[5,10],[15,20]]
    Output: 2
Example 2:

    Input: intervals = [[7,10],[2,4]]
    Output: 1
'''

def meeting_rooms2(self, intervals):

    # thought process: we want to keep a count of the max amount of meetings occuring at the same time. This will give us how many meeting rooms we need. 
    # We can do this by comparing our start times and our end times. Store our start times in 1 array and end times in another. Loop and comapre our values.
    # We want the min between both values, so we move the index of the array with the current min value. 
    # If the min is in the start array, then we have a new starting meeting. Hence we increment our count
    # If the min is in the end array, then we have a finishing meeting. Hence we decrement our count

    n = len(intervals)
    start = []
    end = []

    # create our start and end intervals
    for i in range(n):
        start.append(intervals[i][0])
        end.append(intervals[i][1])

    # sort our arrays
    start.sort()
    end.sort()

    # loop to find our min required meeting count
    start_index = 0
    end_index = 0
    max_count = 0
    curr_count = 0

    while start_index < n and end_index < n:

        if start[start_index] < end[end_index]:
            curr_count += 1
            start_index += 1
        else:
            curr_count -= 1
            end_index += 1
        
        max_count = max(max_count, curr_count)
    
    return max_count


'''
435. Non-overlapping Intervals
Given an array of intervals intervals where intervals[i] = [starti, endi], return the minimum number of intervals you need to remove to make the rest of the intervals non-overlapping.

Note that intervals which only touch at a point are non-overlapping. For example, [1, 2] and [2, 3] are non-overlapping.

Example 1:

    Input: intervals = [[1,2],[2,3],[3,4],[1,3]]
    Output: 1
    Explanation: [1,3] can be removed and the rest of the intervals are non-overlapping.
Example 2:

    Input: intervals = [[1,2],[1,2],[1,2]]
    Output: 2
    Explanation: You need to remove two [1,2] to make the rest of the intervals non-overlapping.
'''
def eraseOverlapIntervals(self, intervals: List[List[int]]) -> int:
    
    # thought process: we want to determine the min number of intervals we need to remove to make the rest of our intervals non-overlapping. We know that the min number of intervals is --> total intervals - max non-overlapping intervals
    # we can get the max non-overlapping interval through using a greedy algorithm by sorting our intervals based on the end times of our intervals. We want to find if the next interval that starts after our current interval

    # sort our intervals based on the end time
    intervals.sort(key=lambda x: x[1])
    
    non_overlapping_intervals = 1
    prev_end = intervals[0][1]

    # we just need to check if the end time is not bigger than start of the next interval
    for interval in intervals:

        # check if the end time is smaller or equal to our next interval start time
        if prev_end <= interval[0]:
            
            # we can increase our count of non overlapping intervals since we found one
            non_overlapping_intervals += 1
            prev_end = interval[1]
    
    # return the total number of intervals - max of non-overlapping intervals 
    return len(intervals) - non_overlapping_intervals


    # Time Complexity: O(n log n)
    # Space Comeplxity: O(1)



'''
297. Serialize and Deserialize Binary Tree
Serialization is the process of converting a data structure or object into a sequence of bits so that it can be stored in a file or memory buffer, or transmitted across a network connection link to be reconstructed later in the same or another computer environment.

Design an algorithm to serialize and deserialize a binary tree. There is no restriction on how your serialization/deserialization algorithm should work. You just need to ensure that a binary tree can be serialized to a string and this string can be deserialized to the original tree structure.

Clarification: The input/output format is the same as how LeetCode serializes a binary tree. You do not necessarily need to follow this format, so please be creative and come up with different approaches yourself.
'''
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

from collections import deque
class Codec:

    def serialize(self, root):
        """Encodes a tree to a single string.
        
        :type root: TreeNode
        :rtype: str
        """
        q = deque()
        q.append(root)

        arr = []

        while q:

            curr_node = q.popleft()

            if curr_node:
                arr.append(str(curr_node.val))
                # append neighbor nodes despite being null
                q.append(curr_node.left)
                q.append(curr_node.right)
            else:
                arr.append("None")


        # **Remove trailing "None" values** (not necessary for reconstruction)
        while arr and arr[-1] == "None":
            arr.pop()

        res = ','.join(arr)

        return res
        

    def deserialize(self, data):
        """Decodes your encoded data to tree.
        
        :type data: str
        :rtype: TreeNode
        """
        # case where we have an empty node
        if not data:
            return None

        # convert our string data to an array
        nodes = data.split(',')

        # create our constants
        n = len(nodes)
        q = deque()

        # create our inital node
        root = TreeNode(nodes[0])
        q.append(root)
        curr_index = 1

        while curr_index < n:

            curr_node = q.popleft()

            # process left node
            if curr_index < n and nodes[curr_index] != "None":
                left = TreeNode(nodes[curr_index])
                curr_node.left = left
                q.append(curr_node.left)
            curr_index += 1

            # process right node
            if curr_index < n and nodes[curr_index] != "None":
                right = TreeNode(nodes[curr_index])
                curr_node.right = right
                q.append(curr_node.right)
            curr_index += 1
        
        return root


    # thought process: use a queue to perform bfs. We can then append to our str our results from our level order traversal. The same process we recreate our nodes through a queue. Turn our string into an array first and keep track of its indices. 

    # Time Complexity: O(n)
    # Space Complexity: O(n)

# Your Codec object will be instantiated and called as such:
# ser = Codec()
# deser = Codec()
# ans = deser.deserialize(ser.serialize(root))


'''
503. Next Greater Element II
Given a circular integer array nums (i.e., the next element of nums[nums.length - 1] is nums[0]), return the next greater number for every element in nums.

The next greater number of a number x is the first greater number to its traversing-order next in the array, which means you could search circularly to find its next greater number. If it doesn't exist, return -1 for this number.

    - THIS IS A MONOTONIC STACK PROBLEM FOR CIRCULAR ARRAYS

Example 1:

    Input: nums = [1,2,1]
    Output: [2,-1,2]
    Explanation: The first 1's next greater number is 2; 
    The number 2 can't find next greater number. 
    The second 1's next greater number needs to search circularly, which is also 2.

'''
def nextGreaterElements(self, nums: List[int]) -> List[int]:
    
    # thought process: we want to find the next greater element of each item. Its important to notice that our array is circular
    
    n = len(nums)
    res = [-1] * n
    stk = []
    
    # since its a circular array, we need to simulate a double pass
    for i in range(2 * n):

        # check if the next item is bigger than our store item (we want only decreasing order in our stack)
        while stk and nums[stk[-1]] < nums[i % n]:
            curr_index = stk.pop()
            res[curr_index] = nums[i % n]
        
        # only append items in our stack once
        if i < n:
            stk.append(i)
    
    return res


'''
547. Number of Provinces
There are n cities. Some of them are connected, while some are not. If city a is connected directly with city b, and city b is connected directly with city c, then city a is connected indirectly with city c.

A province is a group of directly or indirectly connected cities and no other cities outside of the group.

You are given an n x n matrix isConnected where isConnected[i][j] = 1 if the ith city and the jth city are directly connected, and isConnected[i][j] = 0 otherwise.

Return the total number of provinces.

THIS IS A GRAPH PROBLEM THAT PRESENTS US AN ADJACENCY MATRIX
'''
def findCircleNum(self, isConnected: List[List[int]]) -> int:
    
    # thought process: we have 'n' nodes and a matrix desmontrating if the nodes at i and j are connected or not. The goal is to determine how many individual connected components there are. We can do by using union find and connect all of our components. We can also use BFS/DFS to traverse our grid and explore all neighboring cells. Since we know that we have 'n' nodes and that our matrix represents if index i and j are connected, we can loop like so.

    # our dfs helper function for our current task
    def dfs(node, seen, isConnected):
        seen.add(node)

        # now check for that corresponding row all connected nodes to the current one
        for i in range(len(isConnected)):
            if isConnected[node][i] and i not in seen:
                dfs(i, seen, isConnected)

    n = len(isConnected)
    number_of_provinces = 0
    seen = set()

    for i in range(n):
        
        # perform dfs if we have not visited the node yet
        if i not in seen:
            number_of_provinces += 1
            dfs(i, seen, isConnected)
    
    return number_of_provinces
        
    
    # Time Complexity: O(n^2)
    # Space Complexity: O(n)



'''
POPULAR AMAZON QUESTIONS
'''

'''
127. Word Ladder
A transformation sequence from word beginWord to word endWord using a dictionary wordList is a sequence of words beginWord -> s1 -> s2 -> ... -> sk such that:

Every adjacent pair of words differs by a single letter.
Every si for 1 <= i <= k is in wordList. Note that beginWord does not need to be in wordList.
sk == endWord
Given two words, beginWord and endWord, and a dictionary wordList, return the number of words in the shortest transformation sequence from beginWord to endWord, or 0 if no such sequence exists.

 
Example 1:

    Input: beginWord = "hit", endWord = "cog", wordList = ["hot","dot","dog","lot","log","cog"]
    Output: 5
    Explanation: One shortest transformation sequence is "hit" -> "hot" -> "dot" -> "dog" -> cog", which is 5 words long.
Example 2:

    Input: beginWord = "hit", endWord = "cog", wordList = ["hot","dot","dog","lot","log"]
    Output: 0
    Explanation: The endWord "cog" is not in wordList, therefore there is no valid transformation sequence.
'''
from collections import deque
from collections import defaultdict
def ladderLength(self, beginWord: str, endWord: str, wordList: List[str]) -> int:
    
    # thought process: we want to find if its possible to turn our beginWord to our endWord in our dictionary. We want to return the number of words required for the shortest sequence possible. It is possible that beginWord does not exist in our word list. Note that we want a combination of characters, so this hints to backtracking. The reason why we use BFS is that it also hints that it wants the shortest path to our endWord.

    # Possible solution: we can use a BFS and add each word in our queue when we find a valid word. A valid word is a word where only 1 letter differs from the popped word. At each word, we explore all paths that are valid. If we have seen the word, we can skip it. As soon as we find our endWord and its valid from the popped word, then we return its level.

    # we can check possible words by replacing 1 letter with every letter of the alphabet and checking if its in our word list but this is very slow. Instead, make an adjacenecy list where the key represents the combination of any letter with 2 other letters to form our words. 

    # queue variables
    q = deque()
    q.append(beginWord)
    
    # seen set to mark visited nodes
    seen = set()
    seen.add(beginWord)

    # adjacency list to mark patterns to be searched for
    nei = defaultdict(list)
    wordList.append(beginWord)

    # create our adjacency list where each key represents a pattern and the values are all the words that have it
    for word in wordList:
        for i in range(len(word)):
            # modify our original word to change 1 character
            pattern = word[:i] + '*' + word[i+1:]
            nei[pattern].append(word)

    res = 1

    while q:
        
        # process the current level of our BFS tree
        for i in range(len(q)):
            curr_word = q.popleft()

            # case where we found our end word
            if curr_word == endWord:
                return res

            # we now check for all possible patterns caused by switching a letter
            for i in range(len(curr_word)):
                
                pattern = curr_word[:i] + '*' + curr_word[i+1:]

                # check all valid words made from our pattern
                for word in nei[pattern]:
                    if word not in seen:
                        q.append(word)
                        seen.add(word)
        
        # increment our layer as we now search for the next layer
        res += 1

    return 0

'''
200. Number of Islands
Given an m x n 2D binary grid grid which represents a map of '1's (land) and '0's (water), return the number of islands.

An island is surrounded by water and is formed by connecting adjacent lands horizontally or vertically. You may assume all four edges of the grid are all surrounded by water.

Example 1:
    Input: grid = [
    ["1","1","1","1","0"],
    ["1","1","0","1","0"],
    ["1","1","0","0","0"],
    ["0","0","0","0","0"]
    ]
    Output: 1
    Example 2:

'''
def numIslands(self, grid: List[List[str]]) -> int:
    
    # thought process: for each item in our grid, we perform a dfs search to explore all neighboring lands, after visiting we turn it to 0 to mark it as visited
    
    # dfs helper function to explore nearby islands
    def dfs(i, j):

        stk = [(i, j)]

        while stk:

            curr_i, curr_j = stk.pop()

            # check neighboring islands
            for i_off, j_off in [(0,1), (1,0), (0, -1), (-1, 0)]:

                index_i = curr_i + i_off
                index_j = curr_j + j_off

                # mark it as visited and store it in our stack if its valid
                if 0 <= index_i < m and 0 <= index_j < n and grid[index_i][index_j] == '1':
                    grid[index_i][index_j] = '0'
                    stk.append((index_i, index_j))
                

    
    # loop through our grid to check our number of islands
    count = 0
    m = len(grid)
    n = len(grid[0])

    for i in range(m):
        for j in range(n):
            # if we find an island, perform dfs on it
            if grid[i][j] == '1':
                grid[i][j] = '0'
                dfs(i, j)
                count += 1

    return count

    # Time Complexity: O(m * n)
    # Space Complexity: O(m * n)


'''
146. LRU Cache
Design a data structure that follows the constraints of a Least Recently Used (LRU) cache.

Implement the LRUCache class:

LRUCache(int capacity) Initialize the LRU cache with positive size capacity.
int get(int key) Return the value of the key if the key exists, otherwise return -1.
void put(int key, int value) Update the value of the key if the key exists. Otherwise, add the key-value pair to the cache. If the number of keys exceeds the capacity from this operation, evict the least recently used key.
The functions get and put must each run in O(1) average time complexity.
'''
# define a double linked list class
class Node:

    def __init__(self, key=0, val=0):
        self.key = key
        self.val = val
        self.prev = None    
        self.next = None

class LRUCache:

    def __init__(self, capacity: int):
        self.capacity = capacity

        self.cache = {}

        # Have pointer point to the last node and the first node of our linked list
        # note that these are placeholder nodes and serve as only pointers to the first/last items in our lists
        # LRU = LEFT | MRU = RIGHT
        self.right = Node(0, 0)
        self.left = Node(0, 0)
        self.left.next = self.right
        self.right.prev = self.left

    # helper function to help insert node | make our node swaps here 
    def insert(self, node):

        # insert at the MRU position (right)
        prev = self.right.prev
        next = self.right

        prev.next = next.prev = node
        node.next = next 
        node.prev = prev

    # helper function to help remove node | make our node swaps here 
    def remove(self, node):
        node.prev.next = node.next
        node.next.prev = node.prev
      

    def get(self, key: int) -> int:
        
        if key in self.cache:  
            # make current node the most recently used
            self.remove(self.cache[key])
            self.insert(self.cache[key])
            return self.cache[key].val

        return -1


    def put(self, key: int, value: int) -> None:
        
        # Case where we find our node
        if key in self.cache:
            # remove it so we can mark it as the MRU
            self.remove(self.cache[key])
        
        # case where we have to add our pair
        self.cache[key] = Node(key, value)
        self.insert(self.cache[key])

        # case where we have to evict our LRU
        if len(self.cache) > self.capacity:
            
            lru = self.left.next
            self.remove(lru)
            del self.cache[lru.key]

    # Time Complexity: O(1) for all operations
    # Space Complexity: O(n)


# We want to store our cache in a dictionary that points to our doubly linked list containing the chain that maps our cache. We can then have the least recently used (LRU) cache at the end of our doubly linked list and the most recently used one at the start of our queue. 


# Your LRUCache object will be instantiated and called as such:
# obj = LRUCache(capacity)
# param_1 = obj.get(key)
# obj.put(key,value)


'''
42. Trapping Rain Water

Given n non-negative integers representing an elevation map where the width of each bar is 1, compute how much water it can trap after raining.

Input: height = [0,1,0,2,1,0,1,3,2,1,2,1]
Output: 6
Explanation: The above elevation map (black section) is represented by array [0,1,0,2,1,0,1,3,2,1,2,1]. In this case, 6 units of rain water (blue section) are being trapped.
'''
def trap(self, height: List[int]) -> int:
    

    # thought process: we want the total amount of water that can be trapped. Since our bar size have a width of 1, each trapped water area will depend on the height. From observation, water will be trapped when we have a difference in elevation but also when it enclosed between 2 identical or higher heights. For example, although a difference in height of 0 and 1 can cause trapped water, if the area is not enclosed, there will be no water.

    # Attempt 1: we could iterate through our list of heights and at each index, we could look for the next greater height than our current one. We can then increment our count of trapped water by taking the difference of the sequential heights from our current index. We only perform this operation if the previous height is higher than our current one.
    # This turns out to be a brute force solution, returning a O(n^2) time complexity

    # the solution is to use a 2 pointer approach where the left and the right of the current index represents the highest height possible. We then take the min of the two at index i as it represents the potential trapped water. This depends on the current height at index i as well

    l_wall = 0
    r_wall = 0
    n = len(height)

    l_heights = [0] * n
    r_heights = [0] * n

    # store the highest height from the left of our current index and the highest wall from the right of our current index (prefix/suffix arrays)

    for i in range(n):

        # index for right height array
        j = n - i - 1

        # store our current heights
        l_heights[i] = l_wall
        r_heights[j] = r_wall
    
        # store our current highest walls
        l_wall = max(l_wall, height[i])
        r_wall = max(r_wall, height[j])
    
    total_trapped_water = 0
    
    # now loop and get the total trapped water
    for i in range(n):

        # calculate the current trapped water at the current index
        potential_water = min(l_heights[i], r_heights[i]) - height[i]

        # only count our potential water if its a positive number
        if potential_water > 0:
            total_trapped_water += potential_water
    
    
    return total_trapped_water


    # # better solution, using 2 pointers as we can keep track of the left max and right max using variables instead of storing the arrays. We want the min between our left max height and our right max height. Compare heights at all intervals and move pointers when the height is smaller. Since we always want the min of either max left or max right, we only calculate the side that has the current min max height

    # # if our height input is empty, return 0
    # if not height:
    #     return 0

    # l, r = 0, len(height) - 1
    # leftMax, rightMax = height[l], height[r]
    # water = 0

    # while l < r:

    #     # we move our left pointer when the leftMax is smaller (local min)
    #     if leftMax < rightMax:
    #         l += 1
    #         leftMax = max(leftMax, height[l])
    #         water += leftMax - height[l]
    #     else:
    #         r -= 1
    #         rightMax = max(rightMax, height[r])
    #         water += rightMax - height[r]
        
    # return water
            
    
    # # Time Complexity: O(n)
    # # Space Complexity: O(1)
        
'''
3. Longest Substring Without Repeating Characters
Given a string s, find the length of the longest substring without duplicate characters.
Example 1:

    Input: s = "abcabcbb"
    Output: 3
    Explanation: The answer is "abc", with the length of 3.

Example 2:

    Input: s = "bbbbb"
    Output: 1
    Explanation: The answer is "b", with the length of 1.

Example 3:

    Input: s = "pwwkew"
    Output: 3
    Explanation: The answer is "wke", with the length of 3.
    Notice that the answer must be a substring, "pwke" is a subsequence and not a substring.
'''
def lengthOfLongestSubstring(self, s: str) -> int:
    l = 0
    a = set()
    max_substring = 0

    for r in range(len(s)):

        # if our current char is in our set, then remove it, and close the left window
        while s[r] in a:
            a.remove(s[l])
            l += 1

        # add our current char to our set
        a.add(s[r])

        max_substring = max(max_substring, r - l + 1)
    
    return max_substring

    # Time Complexity: O(n)
    # Space Complexity: O(n)

'''
973. K Closest Points to Origin
Given an array of points where points[i] = [xi, yi] represents a point on the X-Y plane and an integer k, return the k closest points to the origin (0, 0).

The distance between two points on the X-Y plane is the Euclidean distance (i.e., √(x1 - x2)2 + (y1 - y2)2).

You may return the answer in any order. The answer is guaranteed to be unique (except for the order that it is in).

Example 1:


    Input: points = [[1,3],[-2,2]], k = 1
    Output: [[-2,2]]
    Explanation:
    The distance between (1, 3) and the origin is sqrt(10).
    The distance between (-2, 2) and the origin is sqrt(8).
    Since sqrt(8) < sqrt(10), (-2, 2) is closer to the origin.
    We only want the closest k = 1 points from the origin, so the answer is just [[-2,2]].
'''
# attempt 2: using a max heap and not using heapify to sort
import heapq
def kClosest(self, points: List[List[int]], k: int) -> List[List[int]]:
    def distance(x, y):
        return x**2 + y**2
    
    max_heap = []

    for x, y in points:

        dist = distance(x,y)

        if len(max_heap) < k:
            heapq.heappush(max_heap, (-dist, x, y))
        else:
            heapq.heappushpop(max_heap, (-dist, x, y))
    
    return [(x,y) for dist, x, y in max_heap]

    # Time Complexity: O(n log k)
    # Space Complexity: O(k)


'''
121. Best Time to Buy and Sell Stock
You are given an array prices where prices[i] is the price of a given stock on the ith day.

You want to maximize your profit by choosing a single day to buy one stock and choosing a different day in the future to sell that stock.

Return the maximum profit you can achieve from this transaction. If you cannot achieve any profit, return 0.

Example 1:

    Input: prices = [7,1,5,3,6,4]
    Output: 5
    Explanation: Buy on day 2 (price = 1) and sell on day 5 (price = 6), profit = 6-1 = 5.
    Note that buying on day 2 and selling on day 1 is not allowed because you must buy before you sell.

Example 2:

    Input: prices = [7,6,4,3,1]
    Output: 0
    Explanation: In this case, no transactions are done and the max profit = 0.
'''
def maxProfit(self, prices: List[int]) -> int: 

    # create a variable to store max profit and lowest price 
    lowest_price = float('inf')
    max_profit = 0

    for price in prices:

        if price < lowest_price:
            lowest_price = price
        
        new_profit = price - lowest_price
        max_profit = max(max_profit, new_profit)
    
    return max_profit

    # Time Complexity: O(n)
    # Space Complexity: O(1)


'''
49. Group Anagrams
Given an array of strings strs, group the anagrams together. You can return the answer in any order.

Example 1:

    Input: strs = ["eat","tea","tan","ate","nat","bat"]
    Output: [["bat"],["nat","tan"],["ate","eat","tea"]]

Explanation:

    There is no string in strs that can be rearranged to form "bat".
    The strings "nat" and "tan" are anagrams as they can be rearranged to form each other.
    The strings "ate", "eat", and "tea" are anagrams as they can be rearranged to form each other.
Example 2:

    Input: strs = [""]
    Output: [[""]]

Example 3:

    Input: strs = ["a"]
    Output: [["a"]]
'''
from collections import defaultdict
def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
    
    # thought process: each string can represent a key in our dictionary
    
    res = defaultdict(list)

    for word in strs:
        key = [0] * 26

        for char in word:
            key[ord(char) - ord('a')] += 1
        
        res[tuple(key)].append(word)

    return [anagram for anagram in res.values()]

    # Time Complexity: O(n)
    # Space Comeplxity: O(26) --> O(1)

'''
23. Merge k Sorted Lists
You are given an array of k linked-lists lists, each linked-list is sorted in ascending order.

Merge all the linked-lists into one sorted linked-list and return it.

Example 1:

    Input: lists = [[1,4,5],[1,3,4],[2,6]]
    Output: [1,1,2,3,4,4,5,6]
    Explanation: The linked-lists are:
    [
    1->4->5,
    1->3->4,
    2->6
    ]
    merging them into one sorted list:
    1->1->2->3->4->4->5->6

Example 2:

    Input: lists = []
    Output: []

Example 3:

    Input: lists = [[]]
    Output: []
'''
import heapq
def mergeKLists(self, lists: List[Optional[ListNode]]) -> Optional[ListNode]:

    # attempt 2: instead of using heapify, we push the node references onto our min heap and keep popping and adding items into it

    # store in our min_heap the node reference and its value | if we have equal value, we can use its index to get the first one
    min_heap = []

    # Push all nodes onto our min heap
    for i, node in enumerate(lists):
        if node:
            heapq.heappush(min_heap, (node.val, i, node))

    dummy = ListNode()
    curr = dummy

    # loop until we popped all our items from our min_heap
    while min_heap:
        
        val, index, node = heapq.heappop(min_heap)

        # Add the next node reference onto our min heap if it exists
        if node.next:
            heapq.heappush(min_heap, (node.next.val, index, node.next))


        # Add our node to our new linked list
        curr.next = node
        curr = curr.next

    return dummy.next

'''
207. Course Schedule
There are a total of numCourses courses you have to take, labeled from 0 to numCourses - 1. You are given an array prerequisites where prerequisites[i] = [ai, bi] indicates that you must take course bi first if you want to take course ai.

For example, the pair [0, 1], indicates that to take course 0 you have to first take course 1.
Return true if you can finish all courses. Otherwise, return false.

 
Example 1:

    Input: numCourses = 2, prerequisites = [[1,0]]
    Output: true
    Explanation: There are a total of 2 courses to take. 
    To take course 1 you should have finished course 0. So it is possible.
Example 2:

    Input: numCourses = 2, prerequisites = [[1,0],[0,1]]
    Output: false
    Explanation: There are a total of 2 courses to take. 
    To take course 1 you should have finished course 0, and to take course 0 you should also have finished course 1. So it is impossible.
'''
from collections import defaultdict
from collections import deque
def canFinish(self, numCourses: int, prerequisites: List[List[int]]) -> bool:
    
    # initial thoughts: our input of number of classes to take is stored in numCourses. The prerequisites input can be transformed to an adjacency list method of storing connected nodes to create a graph, hence a graph problem. Its similar to asking if there is a cycle in a graph, which means in a directed graph, does the current node lead to a node we have already seen.

    # second attempt: numCourses being numbered from 0 to numCourses - 1 indicates the possibility of topological sort starting from 0. With the context of the problem, we can determine that this is a DAG (directed acyclic graph)

    # for DAG, detecting cycles using topological sort is a common algorithm to use. Topological sort is a sorting DAG algorithm that sorts nodes into order. It can be implemented either using the DFS method or the BFS method
    # Time complexity: O(V + E)
    
    # create our indegree array
    indegree = [0] * numCourses

    # creating our adjacency list
    D = defaultdict(list)
    for u, v in prerequisites:
        D[v].append(u)
        indegree[u] += 1
    
    # add all nodes with indegree to 0 into our queue
    q = deque()
    for i in range(numCourses):
        if indegree[i] == 0:
            q.append(i)
    
    # loop through our queue to determine order
    result = []

    while q:

        node = q.popleft()
        result.append(node)
        
        # decrememt the indegree of each adjacent node to our popped one
        for adj in D[node]:
            indegree[adj] -= 1

            # if indegree for current node is 0, then append to queue
            if indegree[adj] == 0:
                q.append(adj)

    
    # return if the length of our result array is equal to numCourses
    return (len(result) == numCourses)

'''
53. Maximum Subarray
Given an integer array nums, find the subarray with the largest sum, and return its sum.

Example 1:

    Input: nums = [-2,1,-3,4,-1,2,1,-5,4]
    Output: 6
    Explanation: The subarray [4,-1,2,1] has the largest sum 6.

Example 2:

    Input: nums = [1]
    Output: 1
    Explanation: The subarray [1] has the largest sum 1.

Example 3:

    Input: nums = [5,4,-1,7,8]
    Output: 23
    Explanation: The subarray [5,4,-1,7,8] has the largest sum 23.
'''
def maxSubArray(self, nums: List[int]) -> int:
    
    # thought process: since its a continuous sub array, we can use kadane's algorithm. Whenever we encounter a negative curr sum, we discard it and reset to 0. This will ensure that we never get a number smaller than 0 (ignore negative numbers)

    max_sum = nums[0]
    curr_sum = 0
    for num in nums:

        curr_sum += num
        max_sum = max(max_sum, curr_sum)

        if curr_sum < 0:
            curr_sum = 0
        
    return max_sum


'''
20. Valid Parentheses
Given a string s containing just the characters '(', ')', '{', '}', '[' and ']', determine if the input string is valid.

An input string is valid if:

Open brackets must be closed by the same type of brackets.
Open brackets must be closed in the correct order.
Every close bracket has a corresponding open bracket of the same type.

Example 1:

    Input: s = "()"
    Output: true

Example 2:

    Input: s = "()[]{}"
    Output: true

Example 3:

    Input: s = "(]"
    Output: false
'''
def isValid(self, s: str) -> bool:
    
    stk = []

    close_to_open = {')' : '(', '}' : '{', ']' : '['}

    for c in s:
        
        # case where we have a closing bracket | only proceed this route if we have items in our stack to process
        if c in close_to_open and stk:

            curr_open_char = stk.pop()

            # evaluate if our close bracket matches with out open bracket
            if curr_open_char != close_to_open[c]:
                return False
        else:
            stk.append(c)

    return True if not stk else False

'''
13. Roman to Integer
Roman numerals are represented by seven different symbols: I, V, X, L, C, D and M.

Symbol       Value
I             1
V             5
X             10
L             50
C             100
D             500
M             1000
For example, 2 is written as II in Roman numeral, just two ones added together. 12 is written as XII, which is simply X + II. The number 27 is written as XXVII, which is XX + V + II.

Roman numerals are usually written largest to smallest from left to right. However, the numeral for four is not IIII. Instead, the number four is written as IV. Because the one is before the five we subtract it making four. The same principle applies to the number nine, which is written as IX. There are six instances where subtraction is used:

I can be placed before V (5) and X (10) to make 4 and 9. 
X can be placed before L (50) and C (100) to make 40 and 90. 
C can be placed before D (500) and M (1000) to make 400 and 900.
Given a roman numeral, convert it to an integer.

 

Example 1:

    Input: s = "III"
    Output: 3
    Explanation: III = 3.

Example 2:

    Input: s = "LVIII"
    Output: 58
    Explanation: L = 50, V= 5, III = 3.
'''
def romanToInt(self, s: str) -> int:

    d = {'I': 1, 'V':5, 'X':10, 'L':50, 'C':100, 'D': 500, 'M':1000}
    summ = 0
    n = len(s)
    i = 0
    
    while i < n:
        if i < n - 1 and d[s[i]] < d[s[i+1]]:
            summ += d[s[i+1]] - d[s[i]]
            i += 2
        else:
            summ += d[s[i]]
            i += 1
    
    return summ
    # Time: O(n)
    # Space: O(1)


'''
127. Word Ladder
A transformation sequence from word beginWord to word endWord using a dictionary wordList is a sequence of words beginWord -> s1 -> s2 -> ... -> sk such that:

Every adjacent pair of words differs by a single letter.
Every si for 1 <= i <= k is in wordList. Note that beginWord does not need to be in wordList.
sk == endWord
Given two words, beginWord and endWord, and a dictionary wordList, return the number of words in the shortest transformation sequence from beginWord to endWord, or 0 if no such sequence exists.

 
Example 1:

    Input: beginWord = "hit", endWord = "cog", wordList = ["hot","dot","dog","lot","log","cog"]
    Output: 5
    Explanation: One shortest transformation sequence is "hit" -> "hot" -> "dot" -> "dog" -> cog", which is 5 words long.
Example 2:

    Input: beginWord = "hit", endWord = "cog", wordList = ["hot","dot","dog","lot","log"]
    Output: 0
    Explanation: The endWord "cog" is not in wordList, therefore there is no valid transformation sequence.
'''
from collections import deque
from collections import defaultdict

def ladderLength(self, beginWord: str, endWord: str, wordList: List[str]) -> int:
    
    # thought process: we want to find if its possible to turn our beginWord to our endWord in our dictionary. We want to return the number of words required for the shortest sequence possible. It is possible that beginWord does not exist in our word list. Note that we want a combination of characters, so this hints to backtracking. The reason why we use BFS is that it also hints that it wants the shortest path to our endWord.

    # Possible solution: we can use a BFS and add each word in our queue when we find a valid word. A valid word is a word where only 1 letter differs from the popped word. At each word, we explore all paths that are valid. If we have seen the word, we can skip it. As soon as we find our endWord and its valid from the popped word, then we return its level.

    # we can check possible words by replacing 1 letter with every letter of the alphabet and checking if its in our word list but this is very slow. Instead, make an adjacenecy list where the key represents the combination of any letter with 2 other letters to form our words. 

    # queue variables
    q = deque()
    q.append(beginWord)
    
    # seen set to mark visited nodes
    seen = set()
    seen.add(beginWord)

    # adjacency list to mark patterns to be searched for
    nei = defaultdict(list)
    wordList.append(beginWord)

    # create our adjacency list where each key represents a pattern and the values are all the words that have it
    for word in wordList:
        for i in range(len(word)):
            # modify our original word to change 1 character
            pattern = word[:i] + '*' + word[i+1:]
            nei[pattern].append(word)

    res = 1

    while q:
        
        # process the current level of our BFS tree
        for i in range(len(q)):
            curr_word = q.popleft()

            # case where we found our end word
            if curr_word == endWord:
                return res

            # we now check for all possible patterns caused by switching a letter
            for i in range(len(curr_word)):
                
                pattern = curr_word[:i] + '*' + curr_word[i+1:]

                # check all valid words made from our pattern
                for word in nei[pattern]:
                    if word not in seen:
                        q.append(word)
                        seen.add(word)
        
        # increment our layer as we now search for the next layer
        res += 1

    return 0

    # Time Comeplxity: O(n * m * 26) | number of words = n and number of char per word = m
    # Space Complexity: O(n * m)



'''
239. Sliding Window Maximum
You are given an array of integers nums, there is a sliding window of size k which is moving from the very left of the array to the very right. You can only see the k numbers in the window. Each time the sliding window moves right by one position.

Return the max sliding window.

Example 1:

Input: nums = [1,3,-1,-3,5,3,6,7], k = 3
Output: [3,3,5,5,6,7]
Explanation: 
Window position                Max
---------------               -----
[1  3  -1] -3  5  3  6  7       3
 1 [3  -1  -3] 5  3  6  7       3
 1  3 [-1  -3  5] 3  6  7       5
 1  3  -1 [-3  5  3] 6  7       5
 1  3  -1  -3 [5  3  6] 7       6
 1  3  -1  -3  5 [3  6  7]      7
Example 2:

Input: nums = [1], k = 1
Output: [1]
'''
from collections import deque
def maxSlidingWindow(self, nums: List[int], k: int) -> List[int]:
    
    # thought process: we can use a naive solution of sliding window by recalculating the max at every step. This will result in a O(n^2) time complexity. Instead, we should use a monotonic queue as we want to maintain the order of max element


    # our monotonic queue will always be in decresing order
    # we use our queue to eliminate all smaller numbers than our max | we can remove elements from the left and the right
    q = deque()        

    l = r = 0
    res = []

    # we want to keep our max at the left side of our queue
    while r < len(nums):
        
        # pop all elements smaller than our current as its not going to be of any value
        while q and nums[q[-1]]< nums[r]:
            q.pop()

        # append our new item
        q.append(r)

        # check if our most left item is not in our window anymore
        if l > q[0]:
            q.popleft()
        
        # make sure that we have a valid window size and append our right most item (which will be the max)
        if (r + 1) >= k:
            res.append(nums[q[0]])
            l += 1
        r += 1
    
    return res

    # Time Complexity: O(n)
    # Space Complexity: O(n)



'''
1209. Remove All Adjacent Duplicates in String II
You are given a string s and an integer k, a k duplicate removal consists of choosing k adjacent and equal letters from s and removing them, causing the left and the right side of the deleted substring to concatenate together.

We repeatedly make k duplicate removals on s until we no longer can.

Return the final string after all such duplicate removals have been made. It is guaranteed that the answer is unique.

 
Example 1:

    Input: s = "abcd", k = 2
    Output: "abcd"
    Explanation: There's nothing to delete.
Example 2:

    Input: s = "deeedbbcccbdaa", k = 3
    Output: "aa"
    Explanation: 
    First delete "eee" and "ccc", get "ddbbbdaa"
    Then delete "bbb", get "dddaa"
    Finally delete "ddd", get "aa"
'''
def removeDuplicates(self, s: str, k: int) -> str:
    
    # thought process: we can use a stack to process the last item stored in it and we can compare if it equal to our currently stored item. We can then pop k times if we reached adjacent duplicates of size k
    
    # store the char and the count of that character
    stk = [] # [char, count]

    for c in s:
        
        # increment our count if we found a matching letter
        if stk and stk[-1][0] == c:
            stk[-1][1] += 1
        
        else:
            stk.append([c, 1])
        
        # if the current letters are of size k, remove them
        if stk[-1][1] == k:
            stk.pop()
    

    # Now build our final solution
    res = ""

    for char, count in stk:

        # add the number of letters needed to build our res string
        res += (char * count)

    
    return res
        
    # Time Complexity: O(n)
    # Space Complexity: O(n)