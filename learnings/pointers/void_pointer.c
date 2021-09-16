#include <stdio.h>
#include <string.h>

int main()
{
    int a = 345;
    float b = 8.3;

    void *ptr;

    // cant be directly dereferenced 
    // printf("%d\n",*ptr); // this will not work
    ptr = &a;
    printf("%d\n", *(int *)ptr);

    ptr = &b;
    printf("%f\n",*(float *)ptr);
    return 0;

}