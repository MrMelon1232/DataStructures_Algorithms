"""
We're going to explore the sliding window algorithm
the demo will be done in python, familiarizing the notion of these data structures through python
"""
# Sliding windows algorithm are good to try when the problem addresses sub arrays or sub strings

# Variable window size
# Problem leetcode #3: find the length of the longest sub string without repeating character
# time complexity: O(n) | space complexity: O(n)
def lengthOfLongestSubstring(s):
    l = 0
    longest = 0
    sett = set()
    n = len(s)

    # O(n)
    for r in range(n):
        # If item is in set, increment to change window and remove item in sett
        while s[r] in sett:
            sett.remove(s[l])
            l+=1
        
        # Calculate longest substring length and take max 
        windowSize = (r - l) + 1
        longest = max(longest, windowSize)

        # Add item in set
        sett.add(s[r])
    
    return longest

# Fixed window size
# Problem leetcode #643: find maximum average value of fixed subarray length k and return it
# time complexity: O(n) | space complexity: O(n)
def findMaxAverage(arr, k):
    curr_sum = 0
    n = len(arr)

    # Calculate initial sum
    for i in range(k):
        curr_sum += arr[i]
    
    # Calculate initial average
    max_avg = curr_sum / k

    # Calculate the sum for new window
    for i in range(k, n):
        curr_sum += arr[i]
        curr_sum -= arr[i - k]
        curr_avg = curr_sum / k
        max_avg = max(max_avg, curr_avg)

    return max_avg

# testing
B = [1, 12, -5, -6, 50, 3]
avg_sol = findMaxAverage(B, 4)
print(avg_sol)
        