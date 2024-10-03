/**
* This is an example of linear search and binary search (considering array is sorted)
*/

//Standard includes
#include <stdio.h>
#include <stdbool.h>

/**
 * @brief Perform linear search on a list and find the specified target
 *
 * @param list list of integers
 * @param target target integer to look for
 * @return true if found false otherwise.
 *
 */
bool linearSearch(int list[], int target)
{
    for(int i = 0 ; i < sizeof(list) ; i++)
    {
        if(list[i] == target)
        {
            printf("Found target at index: %d", i);
            return true;
        }
    }
    return false;
}

/**
 * @brief Perform linear search on a list and find the specified target
 *
 * @param list list of integers
 * @param target target integer to look for
 * @return true if found false otherwise.
 *
 */
bool binarySearch(int list[], int target)
{
    for(int i = 0 ; i < sizeof(list) ; i++)
    {
        if(list[i] == target)
        {
            printf("Found target at index: %d", i);
            return true;
        }
    }
    return false;
}

int main(void)
{
    
}