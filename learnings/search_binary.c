#include <stdio.h>

/** 
 * this is a recursive approach to binary search
 * 
 * @param arr : Array
 * @param low : lower index
 * @param high : higher index
 * @param element : element to find
 **/
int binary_search_recursion(int arr[], int low, int high, int element)
{
    
    int mid;

    // get mid point
    mid = low + (high - low) / 2;

    if (low < high)
        if (arr[mid] == element) // try to find the element at mid point
            return mid;
        else if (element > arr[mid]) // else either left side of mid or right side
            return binary_search_recursion(arr, mid + 1, high, element);
        else
            return binary_search_recursion(arr, low, mid - 1, element);
    else
        return -1;
}

/**
 * this is a iterative approach to binary search 
 * @param arr : Array
 * @param size : size of array
 * @param element : element to find
 **/
int binary_search_iterative(int arr[], int size, int element)
{
    
    int low = 0;
    int high = size - 1;
    int mid;
    while (low <= high)
    {
        mid = low + (high - low) / 2;

        if (arr[mid] == element) // try to find the element at the mid
            return mid;
        if (arr[mid] < element) //
            low = mid + 1;
        else
            high = mid - 1;
    }
    return -1;
}

int main()
{

    int a[] = {1, 2, 3, 4, 5, 6, 7, 8};
    int size = sizeof(a) / sizeof(int);

    printf("%d\n", size);

    int get_index = binary_search_recursion(a, 0, size, 2);

    if (get_index < 0)
        printf("Element not found");
    else
        printf("Element found at : %d", get_index);

    int get_index1 = binary_search_iterative(a, size, 8);

    if (get_index1 < 0)
        printf("Element not found");
    else
        printf("Element found at : %d", get_index1);
    return 0;
}