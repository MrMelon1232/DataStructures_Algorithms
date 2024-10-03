"""
We're going to explore recursion through the fibonacci problem
the demo will be done in python, familiarizing the notion of these data structures through python
"""

"""
Reminder that the fibonacci sequence is the addition of the 2 previous iterations after the 2 base cases of f(0) and f(1)
f(0) = 0
f(1) = 1
therefore f(n) = f(n-2) + f(n-1)
sample output: 0 1 1 2 3 5 8 13 21 ....

When thinking about recursion, think that the algorithm starts from the top and calls the function all the way to the bottom until, we reach a base case
Only then will it start building to reach out final answer
"""
# Time complexity - O(n^2) | Space complexity - O(n)
def fibonacci_brute_recursion(n):
    # Define base cases
    if n == 0:
        return 0
    elif n == 1:
        return 1
    else:
        return fibonacci_brute_recursion(n-1) + fibonacci_brute_recursion(n-2)
    
# Testing
print(fibonacci_brute_recursion(7)) # Should print 13


# Since the computation of the new numbers require the old ones we found, a lot of repetition is made and is a waste of resources
# Therefore, we can use dynamic programming technique called memoization to store the already computed numbers to be reused for the next computations
# Define memo to be reused
# time complexity: O(n) | space complexity: O(n)

memo = {0:0, 1:1}
def fibonacci_memoization(n):
    # Define base cases
    if n in memo:
        return memo[n]
    else:
        memo[n] = fibonacci_memoization(n - 1) + fibonacci_memoization(n - 2)
        return memo[n]
        
print(fibonacci_memoization(7))


# Bottom up approach of dynamic programming | aka Tabulation
# make a table and run from bottom up to the solution
# time complexity: O(n) | space complexity: O(n)
def fib_tab(n):
    if n == 0:
        return 0
    if n == 1:
        return 1
    
    dp = [0] * (n+1)
    dp[0] = 0
    dp[1] = 1

    for i in range(2, n+1):
        dp[i] = dp[i - 2] + dp[i - 1]
    
    return dp[n]

print(fib_tab(7))

# To get constant space, we can keep swapping the prev with the curr value and the curr value with the prev + curr
# this is because the sequence only requires 2 of the previous computations, so theres no need to keep all of them when performing tabulation
def fib_tab2(n):
    if n == 0:
        return 0
    if n == 1:
        return 1
    
    prev = 0
    curr = 1

    for i in range(2, n+1):
        prev, curr = curr, prev + curr
    
    return curr

print(fib_tab2(7))
