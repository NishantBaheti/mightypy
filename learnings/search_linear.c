#include <stdio.h>

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

int main()
{

    int a[] = {1, 2, 3, 4, 5, 6, 7};
    int size = sizeof(a) / sizeof(int);

    int get_index = linear_search(a, 7, size);

    if (get_index < 0)
    {
        printf("Element not found");
    }
    else
    {
        printf("Element found at : %d", get_index);
    }
    return 0;
}