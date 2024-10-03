/**
 * @brief Perform binary search on a list and find the specified target
 *
 * @complexity The time complexity is O(log(n)) and the space complexity is O(1)
 *
 * @param list list of integers
 * @param target target integer to look for
 *
 * @return the index if found -1 otherwise.
 *
 */
function binarySearch(list, target) {
  // Assign left and right pointer to be initially beginning of array
  let left = 0;
  let right = list.length - 1;

  // Check if list is not null or undefined
  if (list !== null || list !== undefined) {
    // Loop only while left is smaller or equal than right
    while (left <= right) {
      // Get the middle of the array
      let middle = Math.floor((left + right) / 2);

      // Check if middle is equal to target
      if (list[middle] === target) {
        return middle;
      }

      //Check if the target is smaller than middle
      if (target < list[middle]) right = middle - 1;
      else left = middle + 1;
    }
  }

  return -1;
}

// Test cases
const validArray = [1, 2, 3, 4, 5, 6];
const emptyArray = [];
const stringArray = ["hello", "welcome", "test", "back", "cheetos"];

console.log(binarySearch(emptyArray, 1)); // returns -1
console.log(binarySearch(stringArray, 1)); // returns -1
console.log(binarySearch(validArray, 3)); // returns 2
console.log(binarySearch(validArray, 10)); // returns -1
console.log(binarySearch(validArray, 4)); // returns 3
