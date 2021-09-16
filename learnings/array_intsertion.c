#include <stdio.h>

#define ARRAY_CAP 100

/*
* arr : array
* index : index to which element needs to be inserted
* element : element to be inserted
* size : size of the array
*/
int insert_element(int arr[], int index, int element, int size)
{
    if (size >= ARRAY_CAP)
    {
        return -1;
    }
    else
    {
        for (int i = size - 1; i >= index; i--)
        {
            arr[i + 1] = arr[i];
        }
        arr[index] = element;
        return index;
    }
}

/*
* arr : array
* size : size of the array
*/
void dislay_elements(int arr[], int size)
{

    for (int i = 0; i < size; i++)
    {
        printf("%d\n", arr[i]);
    }
}

int main()
{

    int a[ARRAY_CAP] = {1, 6, 21, 11, 2};
    int size = 5;

    int result = insert_element(a, 2, 100, size);

    if (result != -1)
    {
        size += 1;
    }

    dislay_elements(a, size);
    return 0;
}