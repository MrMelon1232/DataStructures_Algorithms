"""
We're going to explore a recurring pattern in interview problems called Prefix Sum
the demo will be done in python, familiarizing the notion of these data structures through python
"""

# We will explore 2 problems that uses prefix sum as the overall technique to solve these questions.
# The overall rule of prefix sum is that prefix(i, j) = prefix[j] - prefix[i-1] | from index 0, its just prefix[j]
# we can calculate a prefix sum array performing the following operations at every index of our array:
#
#           prefix[i] = prefix[-1] + nums[i]

# This technique works for non sorted, negative values array, contiguous arrayss

'''
303. Range Sum Query - Immutable
Given an integer array nums, handle multiple queries of the following type:

Calculate the sum of the elements of nums between indices left and right inclusive where left <= right.
Implement the NumArray class:

NumArray(int[] nums) Initializes the object with the integer array nums.
int sumRange(int left, int right) Returns the sum of the elements of nums between indices left and right inclusive (i.e. nums[left] + nums[left + 1] + ... + nums[right]).
'''

class NumArray:

    def __init__(self, nums: List[int]):
        self.nums = nums
        self.prefix = [nums[0]]

        # compute our prefix sum
        for i in range(1, len(nums)):
            self.prefix.append(self.prefix[-1] + nums[i])

    def sumRange(self, left: int, right: int) -> int:
        
        # use prefix sum to calculate our answer
        return self.prefix[right] if left == 0 else self.prefix[right] - self.prefix[left-1] 

        # Time Complexity: O(1) for sumRange function

# Your NumArray object will be instantiated and called as such:
# obj = NumArray(nums)
# param_1 = obj.sumRange(left,right)

'''
560. Subarray Sum Equals K
Given an array of integers nums and an integer k, return the total number of subarrays whose sum equals to k.

A subarray is a contiguous non-empty sequence of elements within an array.
'''

class Solution:
    def subarraySum(self, nums: List[int], k: int) -> int:
        
        # thought process: we can use a prefix sum to compute the prefix at each point. 
        # Since we don't know the location/boundaries of our subarrays that will give us 'k' sum, we can calculate it by performing prefix[j] - k = prefix[i] and use a hashmap to map our prefix sum to its frequency count. 
        # Note that 'k' here is equivalent to our prefix sum in subarray (i,j) or prefix(i,j)
        # Basically the number of times that prefix_sum appears is stored in our hash map. 
        # The goal is to find if theres a prefix such that its equal to taking the difference between our curr_sum and k
        

        prefix_count = {0: 1}
        curr_sum = 0
        count = 0

        # loop through our list of numbers
        for num in nums:
            
            # add our sum to our curr sum
            curr_sum += num

            # take the diff of our prefix 
            diff = curr_sum - k
            
            # increase our count if we find our diff in our prefix_count
            count += prefix_count.get(diff, 0)
            
            # store/increment our prefix at curr_sum, meaning that it exists at least once in our prefix array
            prefix_count[curr_sum] = prefix_count.get(curr_sum, 0) + 1
            
        
        return count
