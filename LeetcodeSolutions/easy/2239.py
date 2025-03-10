'''
2239. Find Closest Number to Zero
Given an integer array nums of size n, return the number with the value closest to 0 in nums. If there are multiple answers, return the number with the largest value.
'''

class Solution:
    def findClosestNumber(self, nums: List[int]) -> int:
        # # Store inital value
        # closestIndex = 0
        # n = len(nums)
        
        # # loop over array
        # for i in range(1, n):
        #     # Compare stored index and new value
        #     if abs(nums[i]) < abs(nums[closestIndex]):
        #         closestIndex = i

        #     # If distance values are equal, then take the bigger number
        #     elif abs(nums[i]) == abs(nums[closestIndex]):
        #         if nums[i] > nums[closestIndex]:
        #             closestIndex = i
        
        # return nums[closestIndex]

        # redo of the original solution
        closestNum = nums[0]

        # loop over our list of numbers
        for num in nums: 
            
            # compare our current number with the closest number stored
            if abs(num) < abs(closestNum):
                closestNum = num

        # take the larger number if there are 2 similar items
        if closestNum < 0 and abs(closestNum) in nums:
            return abs(closestNum)
        else:
            return closestNum