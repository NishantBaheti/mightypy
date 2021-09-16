#include <stdio.h>
#include <stdlib.h>

typedef struct ListNode
{
    int data;
    struct ListNode *next;
} ListNode;

int is_empty(ListNode *top)
{
    if (top == NULL)
    {
        return 1;
    }
    else
    {
        return 0;
    }
}
int is_full(ListNode *top)
{
    ListNode *p = (ListNode *)malloc(sizeof(ListNode));
    if (p == NULL)
    {
        return 1;
    }
    else
    {
        free(p);
        return 0;
    }
}

ListNode *push(ListNode *top, int element)
{
    if (is_full(top))
    {
        printf("stack overflow \n");
    }
    else
    {
        ListNode *new_node = (ListNode *)malloc(sizeof(ListNode));

        new_node->data = element;
        new_node->next = top;
        top = new_node;
        return top;
    }
}
int pop(ListNode **top)
{
    if (is_empty(*top))
    {
        printf("stack underflow \n");
    }
    else
    {
        ListNode *p = *top;
        *top = (*top)->next;
        int val = p->data;
        free(p);
        return val;
    }
}

void traverse_stack(ListNode *ptr)
{
    printf("*----------- Printing Stack -------------------------*\n");
    while (ptr != NULL)
    {
        printf("Element : %d \n", ptr->data);
        ptr = ptr->next;
    };
    printf("*----------------------------------------------------*\n");
}

int peek(ListNode *top, int index)
{
    ListNode *ptr = top;
    for(int i =0;i<=index-1 && ptr!= NULL; i++)
    {
        ptr = ptr->next;
    }
    if(ptr != NULL)
        return ptr->data;
    else
        return -1;
}

int main()
{
    ListNode *top = NULL;

    traverse_stack(top);

    top = push(top, 1);
    top = push(top, 2);
    top = push(top, 3);
    top = push(top, 4);

    traverse_stack(top);

    int val = pop(&top);
    printf("popped value : %d\n",val);
    traverse_stack(top);

    printf("peek at index %d, value %d",1,peek(top,1));
    return 0;
}