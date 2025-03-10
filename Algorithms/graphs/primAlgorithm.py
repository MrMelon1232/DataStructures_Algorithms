"""
We're going to explore Prim's algorithm used to find minimum spanning trees (MST) and determine the total minimum cost to link nodes/points
the demo will be done in python, familiarizing the notion of these data structures through python
"""

# Prim's algorithm is used to build a tree/graph with min distance/min cost between our points using a min_heap.
# Its a greedy algorithm that connects the undirected weighted graph, ensuring no cycles in the smallest weight possible, hence a minimum spanning tree (MST)
import heapq
def mst(points : list[list[int]]): 

    n = len(points)
    seen = set()
    total_cost = 0
    # we will be storing our distance and our index of our points in our min heap
    min_heap = [(0, 0)]
    

    # loop until we have all our nodes connected
    while len(seen) < n:

        dist, i = heapq.heappop(min_heap)

        # if we've seen our popped node, continue
        if i in seen:
            continue
        
        # add to our seen set and add to total cost our distance
        seen.add(i)
        total_cost += dist

        # set variable for our current popped point
        x1 = points[i][0]
        x2 = points[i][1]

        # explore our neigbhor points. By adding to our min heap, we make in sort that only the smallest distance will be processed next
        for j in range(n):
            
            # calculate our distance only if the current node has not yet been visited and add it to our min heap
            if j not in seen:
                y1 = points[j][0]
                y2 = points[j][1]
                nei_dist = abs(x1 - y1) + abs(x2 - y2)

                heapq.heappush(min_heap, (nei_dist, j))

    return total_cost 