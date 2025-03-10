"""
We're going to explore Djikstra's Algorithm
the demo will be done in python, familiarizing the notion of these data structures through python
"""

# Djikstra's Algorithm is an algorithm that aims to find the minimum path that connects all connected nodes in weighted graphs
# Hence, we can also use this to find the min path between 2 nodes.
# Note that it only works for positive weights

'''
743. Network Delay Time
You are given a network of n nodes, labeled from 1 to n. You are also given times, a list of travel times as directed edges times[i] = (ui, vi, wi), where ui is the source node, vi is the target node, and wi is the time it takes for a signal to travel from source to target.

We will send a signal from a given node k. Return the minimum time it takes for all the n nodes to receive the signal. If it is impossible for all the n nodes to receive the signal, return -1.
'''

from collections import defaultdict
import heapq
class Solution:
    def networkDelayTime(self, times: List[List[int]], n: int, k: int) -> int:

        # thought process: djikstra's algorithm is used when we want the min path from node A to node B. In this case, we want to find the minimum time that it takes for all nodes to be connected from the source node. 
        
        # create our adjacenecy matrix from our list of edges
        graph = defaultdict(list)

        for u, v, weight in times:
            graph[u].append((v, weight))


        # we use a min heap to store the nodes and there distance to get there. We store our initial source node
        # Our min heap ensures that we get the minimum weight between our nodes 
        min_heap = [(0, k)]

        # create our dictionary to store the distance/time from our source node A to node B where our key is node B and our value is the distance/time
        min_time = {}

        # loop while we have items in our heap
        while min_heap:

            curr_time, curr_node = heapq.heappop(min_heap)

            # if we already got the minimum, skip the node
            if curr_node in min_time:
                continue

            # mark the node as visited | we store the time it takes to get from source node to the current one
            min_time[curr_node] = curr_time

            # visit neighbors of current popped node
            for visit_node, visit_time in graph[curr_node]:
                
                # push our neighbor nodes onto our min heap only if we have not visited it yet
                if visit_node not in min_time:
                
                    # The nodes of the graphs will be organized by minimum weight
                    heapq.heappush(min_heap, (curr_time + visit_time, visit_node))
                
        # check if all nodes have been reached by our source node
        if len(min_time) == n:
            return max(min_time.values())
        
        return -1 

        # Time complexity: O((V + E) log (V))
        # Spacee Complexity: O(V+E) where V = vertice and E = edges

