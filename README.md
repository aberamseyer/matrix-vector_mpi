# Matrix-Vector MPI

Performs matrix-vector multiplication on the command line using MPI processes. Prompts for dimensions *m* and *n*.

build:

```bash
mpicc -g -Wall -o parallel_mv.o parallel_matxvector.c
```

run: 
```bash
mpiexec -n <number of processes> ./parallel_mv.o
```
