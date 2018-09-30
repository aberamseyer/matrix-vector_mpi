/* File:     parallel_matxvector.c
 *
 * Computes matrix vector multiplication in parallel
 *
 * input:   Number of rows and columns
 * output:  Matrix A, vector x, calculated vector y, and elapsed time
 *
 * compile: mpicc -g -Wall -o parallel_mv.o parallel_matxvector.c
 * run:     mpiexec -n <number of processes> ./parallel_mv.o
 *
 * IT 388 - Introduction to Parallel Processing
 * Illinois State University
 */
#include <stdio.h>  // i/o
#include <stdlib.h> // rand(), malloc()
#include <mpi.h>    // mpi functions

int main(int argc, char* argv[]) {
    int m, n;                       // matrix/vector dimensions
    double* A;                      // matrix
    double* x;                      // vector
    double* y;                      // result vector
    double* local_A;                // rows block scattered from A
    double* local_y;                // results to be collected into y
    double start_time, finish_time; // tracking time elapsed
    int my_rank, num_processes;     // mpi variables
    int i, j, k;                    // counter variables
    int work_per_process = 0;       // rows in local_A
    
    // setup MPI
    MPI_Init(NULL, NULL);

    // Get my process rank
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    // Find out how many processes are being used
    MPI_Comm_size(MPI_COMM_WORLD, &num_processes);

    // read rows and columns from user and send m/n to all other processes
    if (my_rank == 0) {
        printf("Enter the number of rows m:\n");
        scanf("%d", &m);
        printf("Enter the number of columns n:\n");
        scanf("%d", &n);
    } 
    MPI_Bcast(&m, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // validate m is divisible by the number of processes
    if (m % num_processes != 0) {
        if (my_rank == 0)
            printf("Number of processes: %d is not divisible by the matrix dimension entered: %d, exiting..\n", num_processes, m);
        MPI_Finalize();

        return 0;
    } else {
        work_per_process = m / num_processes;
        local_A = malloc(work_per_process*n*sizeof(double));
    }
    
    // x will be visible across all processes
    x = malloc(n*sizeof(double));
    // only a subsection of y needs to be allocated on all processes for work
    local_y = malloc(work_per_process*sizeof(double));
    
    // generate/print matrix A and vector x on process 0
    if (my_rank == 0) {
        A = malloc(m*n*sizeof(double)); // A only needs to exist on process 0
        y = malloc(m*sizeof(double));   // y only needs to exist on process 0
        // Generate matrix A
        printf("\nThis is matrix A[%d,%d]  \n",m,n);
        for (i=0;i<m;i++) {
            for (j=0;j<n;j++) {
                A[i*n + j] = rand() % 20;
                printf("%.0f ", A[i*n+j]);
            }
            printf("\n");
        }
        
        // Generate vector
        printf("\nThis is vector x \n");
        for (k=0;k<n;k++) {
            x[k] = rand() % 20;
            printf("%.0f ", x[k]);
            printf("\n");
        }
    }
    
    // Timing matrix-vector multiplication
    MPI_Barrier(MPI_COMM_WORLD);
    start_time = MPI_Wtime();

    // send blocks to work on to all processes
    MPI_Scatter(A, work_per_process*n, MPI_DOUBLE, local_A, work_per_process*n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    // send x to every process for use in computation
    MPI_Bcast(x, n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    // Computes Matrix-vector multiplication 
    for (i = 0; i < work_per_process; i++) {
        local_y[i] = 0.0;
        for (j = 0; j < n; j++){
            local_y[i] += local_A[i*n + j]*x[j];
            // debug print statement
            // printf("process %d: j=%d, local_y[%d]: %.0f * %.0f = %.0f\n", my_rank, j, i, local_A[i*n+j], x[j], local_y[i]);
        }
    }
    // send results from each process back to process 0
    MPI_Gather(local_y, work_per_process, MPI_DOUBLE, y, work_per_process, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    finish_time = MPI_Wtime();

    // print results on process 0
    if (my_rank == 0) {
        printf("\nThis is y= \n");
        for (i=0;i<m;i++)
            printf("%.0f \n",y[i]);
        printf ("Parallel elapsed time: %f sec\n", finish_time - start_time);
    }

    // MPI cleanup
    MPI_Finalize();

    return 0;
}
