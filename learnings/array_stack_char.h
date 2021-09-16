#include <stdio.h>
#include <stdlib.h>

/**
 * Character type stack based on array
 **/
typedef struct ArrayStack
{
    int total_size;
    int top;
    char *arr;
} ArrayStack;


/**
 * Create a stack with given size
 * 
 * @param size size of the stack
 * 
 * @return stack
 * */
ArrayStack *create_stack(int size)
{
    ArrayStack *stack = (ArrayStack *)malloc(sizeof(ArrayStack));

    stack->total_size = size;
    stack->top = -1;
    stack->arr = (char *)malloc(stack->total_size * sizeof(char));
    return stack;
}

/**
 * Check if stack is empty
 * 
 * @param s ArrayStack
 * 
 * @return 1/0
 **/
int is_empty(ArrayStack *s)
{
    if (s->top == -1)
        return 1;
    else
        return 0;
}

/**
 * Check is stack is full
 * 
 * @param s Array stack pointer
 * **/
int is_full(ArrayStack *s)
{
    if (s->top == s->total_size - 1)
        return 1;
    else
        return 0;
}

int push(ArrayStack *s, char element)
{
    if (is_full(s))
    {
        printf("Stack overflow\n");
        return 0;
    }
    else
    {
        s->top++;
        s->arr[s->top] = element;
        return 1;
    }
}

char pop(ArrayStack *s)
{
    if (is_empty(s))
    {
        printf("stack underflow\n");
        return -1;
    }
    else
    {
        char val = s->arr[s->top];
        s->top = s->top - 1;
        return val;
    }
}


char peek(ArrayStack * s,int index)
{
    int pos = s->top - index + 1;
    if(pos > -1 && pos <= s->top){
        return s->arr[pos];
    }
    else{
        printf("not available at index %d\n",index);
        return -1;
    }
}

char stack_top(ArrayStack *s)
{
    if(is_empty(s))
    {
        return -1;
    }
    else
    {
        return s->arr[s->top];
    }
}

void print_stack(ArrayStack *s)
{
    int index = s->top;
    if(is_empty(s)){
        printf("Empty stack\n");
    }
    else
    {
        printf("*--------------- Printing Stack ----------------------*\n");
        while (index >= 0)
        {
            printf("%c \n", s->arr[index]);
            index--;
        }
        printf("*-----------------------------------------------------*\n");
    }
}
