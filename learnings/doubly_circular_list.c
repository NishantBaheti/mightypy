#include <stdio.h>
#include <stdlib.h>

typedef struct Node
{
    int data;
    struct Node *prev;
    struct Node *next;
} DllNode;

void traverse_doubly_linked_list(DllNode *head)
{
    while (head != NULL)
    {
        printf("Element :%d\n", head->data);
        head = head->next;
    }
    printf("*---------------------------------------------------------------*\n");
}

DllNode *insert_at_first(DllNode *head, int element)
{
    DllNode *new_node = (DllNode *)malloc(sizeof(DllNode));
    new_node->data = element;
    new_node->prev = NULL;
    new_node->next = head;

    return new_node;
}

DllNode *insert_at_last(DllNode *head, int element)
{
    DllNode *new_node = (DllNode *)malloc(sizeof(DllNode));
    new_node->data = element;
    new_node->next = NULL;

    DllNode *ptr = head;
    while (ptr->next != NULL)
    {
        ptr = ptr->next;
    }

    ptr->next = new_node;
    new_node->prev = ptr;
    return head;
}

DllNode *insert_at_index(DllNode *head, int element, int idx)
{
    DllNode *new_node = (DllNode *)malloc(sizeof(DllNode));
    new_node->data = element;
    DllNode *ptr = head;

    for (int cursor = 1; cursor <= (idx - 1); cursor++)
    {
        if (ptr->next == NULL)
        {
            break;
        }
        ptr = ptr->next;
    }

    new_node->next = ptr->next;
    new_node->prev = ptr;

    ptr->next->prev = new_node;
    ptr->next = new_node;

    return head;
}

DllNode *delete_at_first(DllNode *head)
{
    DllNode *ptr = head;

    head = head->next;
    head->next->prev = NULL;

    free(ptr);
    return head;
}

DllNode *delete_at_last(DllNode *head)
{
    DllNode *ptr = head;
    while (ptr->next != NULL)
    {
        ptr = ptr->next;
    }

    ptr->prev->next = NULL;
    free(ptr);
    return head;
}

DllNode *delete_at_index(DllNode *head, int idx)
{
    DllNode *p = head;
    DllNode *q = head->next;

    for (int cursor = 1; cursor <= (idx - 1); cursor++)
    {
        if (q->next == NULL)
        {
            break;
        }
        q = q->next;
        p = p->next;
    }

    p->next = q->next;
    q->next->prev = p;

    free(q);
    return head;
}

int main()
{
    DllNode *head = (DllNode *)malloc(sizeof(DllNode));
    DllNode *second = (DllNode *)malloc(sizeof(DllNode));
    DllNode *third = (DllNode *)malloc(sizeof(DllNode));

    head->data = 7;
    head->next = second;
    head->prev = NULL;

    second->data = 10;
    second->prev = head;
    second->next = third;

    third->data = 14;
    third->prev = second;
    third->next = NULL;

    traverse_doubly_linked_list(head);

    head = insert_at_first(head, 3);
    traverse_doubly_linked_list(head);

    head = insert_at_last(head, 100);
    traverse_doubly_linked_list(head);

    head = insert_at_index(head, 8, 2);
    traverse_doubly_linked_list(head);

    head = delete_at_first(head);
    traverse_doubly_linked_list(head);

    head = delete_at_last(head);
    traverse_doubly_linked_list(head);

    head = delete_at_index(head, 2);
    traverse_doubly_linked_list(head);

    return 0;
}