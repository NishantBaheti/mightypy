#include <stdio.h>
#include <stdlib.h>

typedef struct ArrayStack
{
    int total_size;
    int top;
    int *arr;
} ArrayStack;

ArrayStack *create_stack(int size)
{
    ArrayStack *stack = (ArrayStack *)malloc(sizeof(ArrayStack));

    stack->total_size = size;
    stack->top = -1;
    stack->arr = (int *)malloc(stack->total_size * sizeof(int));
    return stack;
}

int is_empty(ArrayStack *s)
{
    if (s->top == -1)
        return 1;
    else
        return 0;
}

int is_full(ArrayStack *s)
{
    if (s->top == s->total_size - 1)
        return 1;
    else
        return 0;
}

int push(ArrayStack *s, int element)
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

int pop(ArrayStack *s)
{
    if (is_empty(s))
    {
        printf("stack underflow\n");
        return -1;
    }
    else
    {
        int val = s->arr[s->top];
        s->top = s->top - 1;
        return val;
    }
}


int peek(ArrayStack * s,int index)
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
            printf("%d \n", s->arr[index]);
            index--;
        }
        printf("*-----------------------------------------------------*\n");
    }
}

int main()
{
    ArrayStack *stack = create_stack(10);

    print_stack(stack);

    push(stack, 1);
    push(stack, 2);
    push(stack, 3);
    print_stack(stack);

    printf("element at index %d is %d\n",2,peek(stack,2));

    printf("%d popped \n", pop(stack));
    print_stack(stack);
    return 0;
}