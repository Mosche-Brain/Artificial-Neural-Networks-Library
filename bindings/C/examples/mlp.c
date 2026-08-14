//
// Created by jaro on 8/14/26.
//

#include <stdio.h>

#include <yann.h>

int main(void)
{
    yann_context_t* context = NULL;
    if (yann_create_context(&context) != YANN_OK)
    {
        fprintf(stderr, "Failed to create YANN context: %s\n", yann_get_last_error());
        return 1;
    }

    yann_destroy_context(context);
    return 0;
}
