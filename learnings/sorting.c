#include "stdio.h"
#include "stdlib.h"

#define print printf

/**
 * Print array elements.
 * 
 * @param arr array pointer.
 * @param n size of array.
 * **/
void print_array(int *arr, int n)
{
    printf("*------------------------------------- Printing Array -------------------------------------------*\n");
    for (int i = 0; i < n; i++)
    {
        printf("%d |\t", arr[i]);
    }
    printf("\n");
    printf("*------------------------------------------------------------------------------------------------*\n");
}

/**
 * Bubble sort.
 * 
 * @note
 * 1. in every pass compare adjescent elements and sort them out.
 * 
 * @param arr array pointer.
 * @param n size of the array.
 * 
 **/
void bubble_sort(int *arr, int n)
{
    int temp;
    for (int pass = 0; pass < n; pass++)
    {

        for (int i = 0; i <= n - 1 - pass; i++)
        {
            printf("pass : %d element : %d\n", pass + 1, i + 1);
            print_array(arr, n);
            if (arr[i] > arr[i + 1])
            {
                temp = arr[i];
                arr[i] = arr[i + 1];
                arr[i + 1] = temp;
            }
        }
    }
    print_array(arr, n);
}

/**
 * Bubble sort optimized method.
 * 
 * @note
 * Breaking early if any of the pass is not needed to be sorted.
 * 
 * @param arr array pointer.
 * @param n size of the array.
 * 
 **/
void bubble_sort_opt(int *arr, int n)
{
    int temp;
    int is_sorted = 0;
    for (int pass = 0; pass < n; pass++)
    {
        is_sorted = 1;
        for (int i = 0; i <= n - 1 - pass; i++)
        {
            printf("pass : %d element : %d\n", pass + 1, i + 1);
            print_array(arr, n);
            if (arr[i] > arr[i + 1])
            {
                temp = arr[i];
                arr[i] = arr[i + 1];
                arr[i + 1] = temp;
                is_sorted = 0;
            }
        }

        if (is_sorted)
        {
            printf("breaking early");
            return;
        }
    }
    print_array(arr, n);
}

/**
 * Selection sort algorithm.
 * 
 * @note
 * select minimum element and replace it with current index element .
 * 
 * @param arr array pointer.
 * @param n size of array.
 ***/
void selection_sort(int *arr, int n)
{
    int temp, min;
    for (int i = 0; i < n - 1; i++)
    {
        print_array(arr, n);
        // find minimum element's index
        min = i;
        for (int j = i + 1; j < n; j++)
        {
            if (arr[min] > arr[j])
            {
                min = j;
            }
        }
        // select minimum element and replace it with current index element
        temp = arr[i];
        arr[i] = arr[min];
        arr[min] = temp;
    }
    print_array(arr, n);
}

/**
 * Insertion sort algorithm
 * 
 * @note
 * select element in the index range and insert it at the index at sorted position.
 * 
 * @param arr array pointer.
 * @param n size of array.
 ***/
void insertion_sort(int *arr, int n)
{
    int ins_value, i, j;
    for (i = 1; i < n; i++)
    {
        print_array(arr, n);
        ins_value = arr[i]; // store the element of current position
        j = i;
        // iterate to prev indices to see if there is an element that is
        // smaller than the insertion element

        // while loop will break either at the 0th index
        // or it reaches a smaller element
        while (arr[j - 1] > ins_value && j >= 1)
        {
            arr[j] = arr[j - 1];
            j--;
        }
        // insert at the index
        arr[j] = ins_value;
    }

    print_array(arr, n);
}

/**
 * Shell sort(n-gap insertion sort) is extension of insertion sort but with a capability of exchanging
 * values that are far apart.
 * efficient for less than 5000 elements in the array.
 * good choice for repetitive sorting of smaller list.
 * 
 * @note
 * how far apart exchange can happen.
 * get a value of h(apart range) that is less than length of array.
 * 
 * @param arr array pointer.
 * @param n size of array.
 * **/
void shell_sort(int *arr, int n)
{
    int ins_value, i, j, h;
    for (h = 0; h < n / 3; h = 3 * h + 1) // getting value of gap(h) b/w exchange elements
        ;
    for (; h > 0; h = h / 3)
    {
        for (i = h; i < n; i++)
        {
            print_array(arr, n);
            ins_value = arr[i]; // store the element of current position
            j = i;
            // iterate to prev indices to see if there is an element that is
            // smaller than the insertion element

            // while loop will break either at the 0th index
            // or it reaches a smaller element
            while (arr[j - 1] > ins_value && j >= 1)
            {
                arr[j] = arr[j - 1];
                j--;
            }
            // insert at the index
            arr[j] = ins_value;
        }
    }
    print_array(arr, n);
}

/**
 * Merge operation of Merge sort Algorithm.
 * 
 * @note
 * This will take the array in such a way that.
 * 
 *                     |
 *                     |
 *  ___________________|______________________
 * | 4 | 7 | 2 |  11 | 15 | 31  | 25 | 8  | 4 | 
 *  ___________________|______________________
 * Left -->>   left_end| mid -->>         right
 * i -->               | j -->
 * 
 * @param arr array pointer.
 * @param left left index.
 * @param mid mid index.
 * @param right right index.
 * 
 * **/
void merge(int *arr, int left, int mid, int right)
{
    int i = left; // tracker of left side
    int j = mid;  // tracker of mid
    int left_end = mid - 1;
    int size = right - left + 1;
    int *tmp = (int *)malloc(size * sizeof(int)); // temp array pointer
    int tmp_cur = left;                           // tmp array will start from left index

    // start both i and j
    while ((i <= left_end) && (j <= right))
    {
        // if element at i index is smaller/equal to element at j
        // then insert ith element in the tmparray
        if (arr[i] <= arr[j])
        {
            tmp[tmp_cur] = arr[i];
            i++;       // inc i
            tmp_cur++; // inc tmp_cursor
        }
        else
        {
            // else put jth element in tmp array
            tmp[tmp_cur] = arr[j];
            j++;
            tmp_cur++;
        }
    }
    // now rest of the elements left in the first half of the array
    // put them in tmp array
    while (i <= left_end)
    {
        tmp[tmp_cur] = arr[i];
        i++;
        tmp_cur++;
    }

    // or there can be elements in the second half of the array
    // put them in tmp array
    while (j <= right)
    {
        tmp[tmp_cur] = arr[j];
        j++;
        tmp_cur++;
    }

    // copy tmp to arr
    while (left <= right)
    {
        arr[left] = tmp[left];
        left++;
    }
}

/**
 * Merge sort algorithm.
 * 
 * @param arr array pointer.
 * @param left left end index.
 * @param right right end index.
 * **/
void merge_sort(int *arr, int left, int right)
{
    int mid;
    print_array(arr, right - left + 1);

    if (right > left)
    {
        mid = (right + left) / 2; // get mid point

        // divide array in two parts based on mid point
        // and pass them in merge sort recursion for dividation
        merge_sort(arr, left, mid);
        print_array(arr, right - left + 1);

        merge_sort(arr, mid + 1, right);
        print_array(arr, right - left + 1);

        // merge arrays
        merge(arr, left, mid + 1, right);
        print_array(arr, right - left + 1);
    }
    print_array(arr, right - left + 1);
}

/**
 * partition for quick sort.
 * 
 * @note
 *      
 *      low |                                   high
 *       ___|______________________________________
 *      | 4 | 7 | 2 |  11 | 15 | 31  | 1 | 8  | 14 | 
 *       ___|_^l_______________________^r__________
 *     pivot|left -->                       <-- right
 * 
 * 1. increase left index while it is less than pivot value.
 * 2. similarily decrease right index while it is greater than pivot value.
 * 
 *       ___|______________________________________
 *      | 4 | 1 | 2 |  11 | 15 | 31  | 7 | 8  | 14 | 
 *       ___|____^r___ ^l___________________________
 * 
 * 3. swap left with right if left < right.
 * 4. if left > right swap pivot to left.
 *       ___|______________________________________
 *      | 2 | 1 | 4 |  11 | 15 | 31  | 7 | 8  | 14 | 
 *       ___|_____^part____________________________
 * 
 * 5. right is partition index.
 * 
 * @param arr array pointer.
 * @param low low end index.
 * @param high high end index.
 * 
 * @return right paritition index.
 **/
int partition(int *arr, int low, int high)
{
    int left, right, temp, pivot = arr[low];
    left = low + 1;
    right = high;
    while (left < right)
    {
        while (arr[left] <= pivot)
            left++;

        while (arr[right] > pivot)
            right--;
        
        if (left < right)
        {
            temp = arr[left];
            arr[left] = arr[right];
            arr[right] = temp;
        }
    }
    arr[low] = arr[right];
    arr[right] = pivot;
    return right;
}

/**
 * Quick sort algorithm.
 * 
 * @param arr array pointer
 * @param low low end index
 * @param high high end index
 * 
 * **/
void quick_sort(int *arr, int low, int high)
{
    int pivot;
    print_array(arr, high - low + 1);

    if (low < high)
    {
        pivot = partition(arr, low, high);
        quick_sort(arr, low, pivot - 1);
        print_array(arr, high - low + 1);

        quick_sort(arr, pivot + 1, high);
        print_array(arr, high - low + 1);
    }
}

int main()
{
    // int a[] = {3, 33, 512, 5, 64, 100, 1, 4, 45, 87, 131, 43, 65, 98, 654, 123, 623};
    int a[] = {4 , 7 , 2 , 11 , 15 , 31 , 1 ,8 , 14}; 
    int n = sizeof(a) / sizeof(int);

    // bubble_sort(a, n);
    // bubble_sort_opt(a, n);
    // selection_sort(a,n);
    // insertion_sort(a, n);
    // shell_sort(a, n);

    // merge_sort(a, 0, n - 1);
    // very important to pass n-1 here
    // was stuck for two days for extra garbage value in the array
    // I AM STUPID

    quick_sort(a, 0, n - 1);

    return 0;
}