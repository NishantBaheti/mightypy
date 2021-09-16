#include <stdio.h>
#include <stdlib.h>

/*************************************************
 * Linked list node structure.
 * 
 * @note Ithas two components
 * data, pointer to the next node in the list
 *************************************************/
typedef struct ListNode
{
    int data;
    struct ListNode *next;
} ListNode;

/*******************************************************
 * traverse circular linked list.
 * 
 * @note run a loop till the pointer to the head doen't reach
 * till the last node (pointing to head)
 *  
 * @param ptr: pointer to the node in the linked list.
 ******************************************************/
void traverse_circular_linked_list(ListNode *head)
{
    ListNode *ptr = head;
    do
    {
        printf("Element : %d\n", ptr->data);
        ptr = ptr->next;
    } while (ptr != head);
    printf("*----------------------------------------------------------*\n");
}

/********************************************************************************
 * Insert element at the first position(at head) of the circular linked list.
 *  
 * @note 
 * 1. create a node with element value.
 * 2. point it to the head node of the cll.
 * 3. create a ptr pointer to the head.
 * 4. run a do while loop till increase ptr's position, ptr's next is not eq to head.
 * 5. point ptr's next to new node as ptr is at a position before head.
 * 
 * @param head: head pointer of the ListNode type repr of the cll.
 * @param element: element that needs to be inserted at the first.
 * 
 * @returns new_node: new node is the head now.
 * 
 *******************************************************************************/
ListNode *insert_at_first(ListNode *head, int element)
{
    ListNode *new_node = (ListNode *)malloc(sizeof(ListNode));
    new_node->data = element;
    new_node->next = head;
    ListNode *ptr = head;
    do
    {
        ptr = ptr->next;
    } while (ptr->next != head);

    ptr->next = new_node;

    return new_node;
}

/********************************************************************************
 * Insert element at the last position of the cll(before head.)
 * 
 * @note
 * 1. create a node with element value and put head's node in the next.
 * 2. create a pointer ptr and point to the head.
 * 3. traverse it to the list till reach at the end(next is head).
 * 4. point ptr's node's next to the new node.
 * 
 * @param head: head pointer of the ListNode type repr of the linked list.
 * @param element: element that needs to be inserted at the last.
 * 
 * @returns head: head pointer.
 *
 *******************************************************************************/
ListNode *insert_at_last(ListNode *head, int element)
{
    ListNode *new_node = (ListNode *)malloc(sizeof(ListNode));
    new_node->data = element;
    new_node->next = head;
    ListNode *ptr = head;
    do
    {
        ptr = ptr->next;
    } while (ptr->next != head);

    ptr->next = new_node;

    return head;
}

/********************************************************************************
 * insert element at a index in cll.
 * 
 * @note
 * 1. create a node of linked list with element value.
 * 2. create a pointer ptr and point to the head.
 * 3. traverse it to the list till reach at index-1 or next is head.
 * 4. point new node's next to the ptr's node's next.
 * 5. ptr's next to the new node.
 * 
 * @param head: head pointer of the ListNode type repr of the ll.
 * @param element: element that needs to be inserted at the first.
 * @param idx: index.
 * 
 * @return head: head of the linked list.
 *******************************************************************************/
ListNode *insert_at_index(ListNode *head, int element, int idx)
{
    ListNode *new_node = (ListNode *)malloc(sizeof(ListNode));
    new_node->data = element;
    ListNode *ptr = head;

    for (int cursor = 1; cursor <= (idx - 1); cursor++)
    {
        if (ptr->next == head)
        {
            break;
        }
        ptr = ptr->next;
    }
    new_node->next = ptr->next;
    ptr->next = new_node;
    return head;
}

/********************************************************************************
 * delete element at the first position of the cll.
 * 
 * @note
 * 1. create a p pointer to head pointing node.
 * 2. create a q pointer to head's next.
 * 3. traverse q to the list till q's node's next is head.
 * 4. q's node's next point to p's node's next.  
 * 4. free up p.
 * 
 * @param head: head pointer of the ListNode type repr of the linked list.
 * 
 * @returns q->next. 
 *******************************************************************************/
ListNode *delete_at_first(ListNode *head)
{
    ListNode *p = head;
    ListNode *q = head->next;

    while (q->next != head)
    {
        q = q->next;
    }
    q->next = p->next;
    free(p);
    return q->next;
}

/********************************************************************************
 * delete element at the last position of the cll.
 * 
 * @note
 * 1. create a pointer p to head.
 * 2. create a pointer q to head's next.
 * 3. traverse until q's node's next is head simultaneously traverse with p as well.
 * 4. p's next eq to q's next.
 * 5. free up q's node.
 * 
 * @param head: head pointer of the ListNode type repr of the cll.
 * 
 * @returns head: new head node pointer of the cll type of ListNode. 
 *******************************************************************************/
ListNode *delete_at_last(ListNode *head)
{
    ListNode *p = head;
    ListNode *q = head->next;

    while (q->next != head)
    {
        p = p->next;
        q = q->next;
    }

    p->next = q->next;
    free(q);
    return head;
}

/********************************************************************************
 * delete element at the indexed position of the cll.
 * 
 * @note 
 * 1. create a pointer p to head.
 * 2. create a pointer q to head's next.
 * 3. traverse linked list till cursor=1 to index-1 or q's next is head. update p and q with next.
 * 4. point p's next to q's next.
 * 5. free q.
 * 
 * @param head: head pointer of the ListNode type repr of the cll.
 * @param idx: index
 * 
 * @returns head.
 *******************************************************************************/
ListNode *delete_at_index(ListNode *head, int idx)
{
    ListNode *p = head;
    ListNode *q = head->next;

    for (int cursor = 1; cursor <= (idx - 1); cursor++){

        if(q->next == head){
            break;
        }
        p = p->next;
        q = q->next;
    }

    p->next = q->next;
    free(q);
    return head;
}

int main()
{
    ListNode *head = (ListNode *)malloc(sizeof(ListNode));
    ListNode *second = (ListNode *)malloc(sizeof(ListNode));
    ListNode *third = (ListNode *)malloc(sizeof(ListNode));

    head->data = 7;
    head->next = second;

    second->data = 10;
    second->next = third;

    third->data = 14;
    third->next = head;

    traverse_circular_linked_list(head);

    head = insert_at_first(head, 3);
    traverse_circular_linked_list(head);

    head = insert_at_last(head, 100);
    traverse_circular_linked_list(head);

    head = insert_at_index(head, 8, 2);
    traverse_circular_linked_list(head);

    head = delete_at_first(head);
    traverse_circular_linked_list(head);

    head = delete_at_last(head);
    traverse_circular_linked_list(head);

    head = delete_at_index(head,2);
    traverse_circular_linked_list(head);

    return 0;
}