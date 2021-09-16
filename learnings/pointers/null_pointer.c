#include <stdio.h>
#include <string.h>

int main(){
    int *ptr = NULL; // null pointer

    // printf("%d", *ptr); // this will fail core dumped -> cant dereference a NULL pointer
    int x;
    x = 10;
    ptr = &x;
    return 0;
}