#include "stdio.h"
#include "string.h"
#include "array_stack_char.h"

int is_operator(char c)
{
    if (c == '/' || c == '*' || c == '+' || c == '-')
    {
        return 1;
    }
    else
    {
        return 0;
    }
}

int precedence(char c)
{
    if (c == '/' || c == '*')
    {
        return 3;
    }
    else if (c == '+' || c == '-')
    {
        return 2;
    }
    else
    {
        return 0;
    }
}


/**
 * Convert infix to postfix expression
 * 
 * @param infix character pointer to infix character string
 * 
 * @returns postfix character pointer to postfix string
 **/
char *infix_to_postfix(char *infix)
{   
    // we need one stack
    ArrayStack *stack_ptr = create_stack(100);
    
    // one postfix character pointer size of infix string
    char *postfix = (char *)malloc((strlen(infix) + 1) * sizeof(char));

    // infix index tracker
    int infix_tracker = 0;

    // postfix index tracker
    int postfix_tracker = 0;

    // running a loop till we reach at the end of the infix string
    while (infix[infix_tracker] != '\0')
    {
        if (!is_operator(infix[infix_tracker]))
        {
            // if character is not an operator add it
            // in the postfix string
            postfix[postfix_tracker] = infix[infix_tracker];
            
            // increment indices
            infix_tracker++;
            postfix_tracker++;
        }
        else
        {
            // if operator check precedence
            if (precedence(infix[infix_tracker]) > precedence(stack_top(stack_ptr)))
            {
                // if infix character precedence is greater than the character present in the stack top
                // then push it in the stack
                push(stack_ptr, infix[infix_tracker]);
                infix_tracker++;
            }
            else
            {
                // or pop it from the stack and add it in postfix string
                postfix[postfix_tracker] = pop(stack_ptr);
                postfix_tracker++;
            }
        }
    }
    while (is_empty(stack_ptr))
    {
        // for rest of the stack operators 
        // pop and add to postfix string
        postfix[postfix_tracker] = pop(stack_ptr);
        postfix_tracker++;
    }
    postfix[postfix_tracker] = '\0';
    return postfix;
}

int main()
{
    char *infix = "x-y/z-k*d";
    printf("postfix is %s", infix_to_postfix(infix));
    return 0;
}