#include "array_stack_char.h"
#include "string.h"

#define MAX_SIZE 1000

/**
 * Match characters for parenthesis
 * 
 * @param a char 1
 * @param b char 2
 * 
 * @return 1/0
 **/
int match(char a, char b)
{
    if (a == '{' && b == '}')
    {
        return 1;
    }
    else if (a == '(' && b == ')')
    {
        return 1;
    }
    else if (a == '[' && b == ']')
    {
        return 1;
    }
    else
    {
        return 0;
    }
}


/**
 * Match paranthesis
 * 
 * @param expr expression character pointer
 * 
 * @return 1/0
 **/
int parenthesis_match(char *expr)
{
    ArrayStack *stack = create_stack(MAX_SIZE);
    char popped_char;
    for (int i = 0; expr[i] != '\0'; i++) // \0 means end of the character string
    {
        if (expr[i] == '(' || expr[i] == '{' || expr[i] == '[')
        {
            push(stack, expr[i]);
        }
        else if (expr[i] == ')' || expr[i] == '}' || expr[i] == ']')
        {
            if (is_empty(stack))
            {
                return 0;
            }
            else
            {
                popped_char = pop(stack);
                if (!match(popped_char, expr[i]))
                {
                    return 0;
                }
            }
        }
    }
    if (is_empty(stack))
    {
        return 1;
    }
    else
    {
        return 0;
    }
}

int main()
{
    char *expr = "((8) *(9)[])";

    if (parenthesis_match(expr))
        printf("correcto");
    else
        printf("error");
    return 0;
}