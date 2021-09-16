#include "stdio.h"
#include "stdlib.h"

typedef struct Queue
{
    int total_size;
    int front;
    int rear;
    int *arr;
} Queue;

Queue *create_queue(int size)
{
    Queue *q_ptr = (Queue *)malloc(sizeof(Queue));
    q_ptr->total_size = size;
    q_ptr->front = -1;
    q_ptr->rear = -1;
    q_ptr->arr = (int *)malloc(size * sizeof(int));
    return q_ptr;
}

int is_empty(Queue *q)
{
    if (q->front == q->rear)
    {
        return 1;
    }
    else
    {
        return 0;
    }
}

int is_full(Queue *q)
{
    if (q->rear == q->total_size - 1)
    {
        return 1;
    }
    else
    {
        return 0;
    }
}

void enqueue(Queue *q, int elememt)
{
    if (is_full(q))
    {
        printf("The queue is full.\n");
    }
    else
    {
        // increase rear index and insert element at that index
        q->rear++;
        q->arr[q->rear] = elememt;
        printf("Enqueued element : %d \n", elememt);
    }
}

int dequeue(Queue *q)
{
    if (is_empty(q))
    {
        printf("Queue is empty.\n");
        return -1;
    }
    else
    {
        // increase the front index and return
        q->front++;
        int a = q->arr[q->front];
        return a;
    }
}

void print_queue(Queue *q)
{
    printf("*------------------ Queue -----------------------------*\n");
    for (int i = q->front+1; i <= q->rear; i++)
    {
        printf("Element : %d\n", q->arr[i]);
    }
    printf("*------------------------------------------------------*\n");
}

int main()
{
    Queue *q_ptr = create_queue(100);

    printf("dequeued element : %d\n", dequeue(q_ptr));

    enqueue(q_ptr, 10);
    enqueue(q_ptr, 20);
    enqueue(q_ptr, 30);
    enqueue(q_ptr, 40);

    print_queue(q_ptr);

    printf("dequeued element : %d\n",dequeue(q_ptr));

    print_queue(q_ptr);
    return 0;
}