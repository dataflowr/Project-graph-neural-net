#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <mpi.h>

int rank, size;

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    struct timeval t1, t2;
    gettimeofday(&t1, NULL);

    system("python3 commander.py train");

    gettimeofday(&t2, NULL);

    double duration = (t2.tv_sec - t1.tv_sec) + ((t2.tv_usec - t1.tv_usec) / 1e6);

    if (rank==0){
    printf("Simulation took %lf s to complete\n", duration);
    }

    MPI_Finalize();
    return 0;
}
