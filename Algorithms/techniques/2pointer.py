"""
We're going to explore the 2 pointer algorithm aka squeeze
the demo will be done in python, familiarizing the notion of these data structures through python
"""

# Algorithm
# Problem Leetcode #977: given an integer array nums sorted in increasing order, return an array of the square of each number sorted in increasing order
# catch: note that negative numbers squared might be the highest now
# logic, since we square all numbers, logically, the biggest number is either at the end of the array or the start
# time complexity: O(log n) | Space complexity: O(n)

def sortedsquares(nums):
    left = 0
    right = len(nums) - 1
    result = []

    while left <= right:
        if abs(nums[left]) > abs(nums[right]):
            result.append(nums[left] ** 2)
            left+=1
        else:
            result.append(nums[right] ** 2) 
            right-=1
    
    result.reverse()

    return result

# testing
A = [-4, -1, 0, 3, 10]
B = sortedsquares(A)
print(B)


# note that the other approach is to brute force it by squaring it right away, then sorting using one of the sorting algorithm