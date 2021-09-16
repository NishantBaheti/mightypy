#include <stdio.h>
#include <stdlib.h>

/* 
* total_size : total size of the array
* used_size : used size of the array
* ptr : pointer to the index address
*/
typedef struct MyArray
{

    int total_size;
    int used_size;
    int *ptr;
} Array;

/**
 * @param a : pointer the the array (when we want value of an address we use *)
 * @param total_size : total size to the array
 * @param used_size : size we want to use now    
 **/
void create_array(Array *a, int total_size, int used_size)
{

    // noob approach
    // (*a).total_size = total_size;
    // (*a).used_size = used_size;
    // (*a).ptr = (int *)malloc(total_size*sizeof(int));

    a->total_size = total_size;
    a->used_size = used_size;
    a->ptr = (int *)malloc(total_size * sizeof(int));
}

/**
 * @param a : pointer the the array (when we want value of an address we use *)
 **/
void show(Array *a)
{

    for (int i = 0; i < a->used_size; i++)
    {
        printf("%d\n", (a->ptr)[i]);
    }
}

/**
 * @param a : pointer the the array (when we want value of an address we use *)
 **/
void set_values(Array *a)
{

    int n;
    for (int i = 0; i < a->used_size; i++)
    {

        printf("Enter element for index: %d\n", i);
        scanf("%d", &n);
        (a->ptr)[i] = n;
    }
}

int main()
{
    Array marks;
    create_array(&marks, 20, 2); // passing the address of Array marks so pointer can catch it
    set_values(&marks);
    show(&marks);
    return 0;
}