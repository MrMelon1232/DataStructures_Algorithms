/**
 * @brief implementation of graph structure
 *
 * @attribute nbOfVertices number of vertices in the
 * @attribute adjList list of all vertices that are connected
 *            key represents the vertice and the value is a list of all connected vertices
 *
 */
class Graph {
  constructor(nbOfVertices) {
    this.nbOfVertices = nbOfVertices;
    this.adjList = new Map();
  }

  /**
   * @brief function to add vertices to graph
   *
   * @param v vertex to be added
   *
   */
  addVertice(v) {
    this.adjList.set(v, []);
  }

  /**
   * @brief function to add vertices to graph
   *
   * @param v vertex to be added
   *
   */
  addEdge(v, w) {
    // add vertice connection from v to w
    this.adjList.get(v).push(w);

    // add vertice connection from w to v
    this.adjList.get(w).push(v);
  }
}

/**
 * @brief DFS for a graph includes checking for cycles
 *
 * @param node node passed down in the function
 *
 */
function dfs(list, nbOfVertices, startingNode) {
  if (list === null) return -1;

  // initialize stack
  var stack = [];

  // initialize visited array
  var visited = Array(nbOfVertices).fill(false);

  // initialize current item
  var adjacentNodes;

  // Store initial root node in stack
  stack.push(list.get(startingNode));

  // Loop while we have items in the stack
  while (stack.length > 0) {
    // pop current node and print
    adjacentNodes = stack.pop();

    // store linked nodes to the one we popped
    for (var i = 0; i < adjacentNodes.length; i++) {
      // check if node has been visited, if not then add it to the stack
      if (!visited[i]) {
        stack.push(adjacentNodes[i]);
        visited[i] = true;
      }
    }
  }
}

// Create our graph
var graph = new Graph();
var vertices = ["A", "B", "C", "D", "E", "F", "G", "H"];

// Adding the vertices
for (var i = 0; i < vertices.length; i++) {
  graph.addVertice(vertices[i]);
}

// Add the connection between the vertices (edges)
/**
 * Here is a smalld description of how our graph looks like:
 *
 *  A ------ B
 *  | --     |
 *  |   |    |
 *  |    --  |
 *  C      |-D
 *  |
 *  |
 *  E -------F
 *  |
 *  |
 *  G -------H
 */
graph.addEdge("A", "B");
graph.addEdge("A", "C");
graph.addEdge("A", "D");
graph.addEdge("B", "D");
graph.addEdge("C", "E");
graph.addEdge("E", "F");
graph.addEdge("E", "G");
graph.addEdge("G", "H");

dfs(graph.adjList, graph.nbOfVertices, "A");
