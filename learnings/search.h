#include "stdio.h"

/** 
 * @param arr : Array
 * @param size : size of array
 * @param element : element to find
 * 
 * @returns : index or -1
 **/
int linear_search(int arr[], int size, int element)
{

    for (int i = 0; i < size; i++)
    {
        if (arr[i] == element)
        {
            return i;
        }
    }
    return -1;
}

/** 
 * this is a recursive approach to binary search
 * 
 * @param arr : Array
 * @param low : lower index
 * @param high : higher index
 * @param element : element to find
 * 
 * @return index or -1(not found)
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
 * 
 * @return index or -1
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