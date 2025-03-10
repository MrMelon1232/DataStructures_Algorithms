"""
We're going to explore a greedy algorithm called Kadane's Algorithm used in dynamic programming
the demo will be done in python, familiarizing the notion of these data structures through python
"""

# Kadane's Algorithm is a greedy search algorithm and DP one used to find a local solution to find our global solution
# this pattern can be used in multiple problems, we will explore 2 of them.

'''
560. Subarray Sum Equals K
Given an array of integers nums and an integer k, return the total number of subarrays whose sum equals to k.

A subarray is a contiguous non-empty sequence of elements within an array.

Example 1:

    Input: nums = [-2,1,-3,4,-1,2,1,-5,4]
    Output: 6
    Explanation: The subarray [4,-1,2,1] has the largest sum 6.
'''
class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        
        # thought process: since its a continuous sub array, we can use kadane's algorithm. Whenever we encounter a negative curr sum, we discard it and reset to 0. 
        # This will ensure that we never get a number smaller than 0 (ignore negative numbers)

        curr_sum = 0
        max_sum = nums[0]

        for num in nums:

            curr_sum += num
            max_sum = max(max_sum, curr_sum)

            # in the case that we have a curr_sum smaller than 0, reset it back to 0
            if curr_sum < 0:
                curr_sum = 0
            
        
        return max_sum
    

'''
169. Majority Element
Given an array nums of size n, return the majority element.

The majority element is the element that appears more than ⌊n / 2⌋ times. You may assume that the majority element always exists in the array.

Example 1:

    Input: nums = [3,2,3]
    Output: 3

Example 2:

    Input: nums = [2,2,1,1,1,2,2]
    Output: 2
'''

from collections import Counter 
class Solution:
    def majorityElement(self, nums: List[int]) -> int:
        
        # thought process: we can use a counter to keep track of the occurence of each element in our list of numbers. Then we can take the max between all counted values and return it. 

        # The 2nd method we can use is linearly going through our array, we can store our majority element and increment a count every time we see it. When we see a different number, we can decrement the count. When our count reaches 0, we therefore have a new majority element.


        # method 1: O(n) but O(n) space
        # c = Counter(nums)
        # majority_element = max(c, key=c.get)
        # return majority_element

        # method 2: O(n) and O(1) space
        majority_element = nums[0]
        counter = 1
        
        for num in nums:

            # case where we find our majority element
            if num == majority_element:
                counter += 1
            else:

                # case where we find a different number than our majority element
                counter -= 1

                # case where we found a new majority element
                if counter == 0:
                    majority_element = num
                    counter += 1
        
        return majority_element