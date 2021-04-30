#ifndef utilities_cu_
#define utilities_cu_

#include <stdio.h>

void prettyPrint(float *m, unsigned int vectors, unsigned int values) {
  if (m == NULL) {
    printf("<null>\n");
    return;
  }

  if (vectors <= 0 || values <= 0) {
    printf("<null[%u][%u]>\n", vectors, values); 
    return;
  }

  for (int vector = 0; vector < vectors; vector++) {
    for (int value = 0; value < values; value++)
      printf("%05.1f ", m[vector * values + value]);
    printf("\n");
  }
  printf("\n");
}
#endif
