/**
 * @brief Perform depth first search on a tree and find the specified target.
 *        We will also explore the multiple ways to traverse trees.
 *
 * @complexity The time complexity is O(n) and the space complexity is O(log(n))
 */

/**
 * @brief Class for a node structure
 *
 * @attribute value value of current node
 * @attribute left left node connected to current one
 * @attribute right right node connected to current one
 *
 */

class Node {
  constructor(value) {
    this.value = value;
    this.left = null;
    this.right = null;
  }
}

/**
 * @brief Pre order traversal of a binary tree
 * (top to bottom but left to right | dfs)
 *
 * @param node node passed down in the function
 *
 */
function preOrder(node) {
  //Check if node is null
  if (node === null) {
    return;
  }

  console.log(node.value);
  preOrder(node.left);
  preOrder(node.right);
}

// Creating our tree
/**
 * Note our structure ressembles this drawing
 *
 *       1
 *   2       3
 * 4   5       6
 */
var root = new Node(1);
root.left = new Node(2);
root.right = new Node(3);
root.left.left = new Node(4);
root.left.right = new Node(5);
root.right.left = new Node(6);

// Testing preOrder traversal
console.log("Testing preOrder traversal method:");
preOrder(root); // output: 1 2 4 5 3 6

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/**
 * @brief In order traversal of a binary tree
 * (bot to top but left to right)
 *
 * @param node node passed down in the function
 *
 */
function inOrder(node) {
  //Check if node is null
  if (node === null) {
    return;
  }

  inOrder(node.left);
  console.log(node.value);
  inOrder(node.right);
}

// Testing inOrder traversal
console.log("\nTesting inOrder traversal method:");
inOrder(root); // output: 4 2 5 1 6 3

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/**
 * @brief Post order traversal of a binary tree
 * (bot to top but left to right | bottom to top more important)
 *
 * @param node node passed down in the function
 *
 */
function postOrder(node) {
  //Check if node is null
  if (node === null) {
    return;
  }

  postOrder(node.left);
  postOrder(node.right);
  console.log(node.value);
}

// Testing postOrder traversal
console.log("\nTesting postOrder traversal method:");
postOrder(root); // output: 4 5 2 6 3 1

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/**
 * @brief BFS or level traversal of a binary tree
 * top to bottom | visit all nodes closest to root first
 *
 * can be implemented using a queue
 *
 * @param node node passed down in the function
 *
 */
function bfs_traversal(node) {
  // Check if node is null
  if (node === null) {
    return -1;
  }

  // Create our queue item
  var queue = [];

  // Store the root node in queue
  queue.push(node);

  // Loop for printout level traversal of tree
  while (queue.length > 0) {
    // Add left and right node of current stored node
    // Add to queue left node
    if (queue[0].left !== null) queue.push(queue[0].left);

    // Add to queue right node
    if (queue[0].right !== null) queue.push(queue[0].right);

    // Retrieve and print front node
    console.log(queue.shift().value);
  }
}

// Testing bfs_traversal traversal
console.log("\nTesting bfs_traversal traversal method:");
bfs_traversal(root); // 1 2 3 4 5 6

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
