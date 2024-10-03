/**
 * @brief Perform linear search on a list and find the specified target
 *
 * @complexity The time complexity is O(n) and the space complexity is O(1)
 *
 * @param list list of integers
 * @param target target integer to look for
 * @return the index if found -1 otherwise.
 *
 */
function linearSearch(list, target) {
  // Check if list is not null or undefined
  if (list !== null || list !== undefined) {
    for (let i = 0; i < list.length - 1; i++) {
      if (list[i] === target) return i;
    }
  }

  return -1;
}

// Test cases
const validArray = [1, 2, 3, 4, 5];
const emptyArray = [];
const stringArray = ["hello", "welcome", "test", "back", "cheetos"];

console.log(linearSearch(emptyArray, 1)); // returns -1
console.log(linearSearch(stringArray, 1)); // returns -1
console.log(linearSearch(validArray, 3)); // returns 2
console.log(linearSearch(validArray, 10)); // returns -1
