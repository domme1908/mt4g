#ifndef UTILS_H
#define UTILS_H

#include <cstdlib>

static inline void fisher_yates_shuffle(unsigned int *array, int n)
{
    for (int i = n - 1; i > 0; i--)
    {
        int j = rand() % (i + 1);
        unsigned int temp = array[i];
        array[i] = array[j];
        array[j] = temp;
    }
}

#endif
