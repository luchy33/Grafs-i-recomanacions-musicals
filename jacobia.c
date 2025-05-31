#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

#define m 4096          // columnas fijas
#define iter_max 1000   // iteraciones fijas
#define tol 3.0e-3f

int main(int argc, char* argv[]) {
    int n;
    if (argc > 1) n = atoi(argv[1]); // n pasado como argumento (número de filas)

    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const float pi = 2.0f * asinf(1.0f);
    float error = 1.0f;
    int iter = 0;

    // Dividir filas entre procesos
    int local_n = n / size;
    int start = rank * local_n;
    int end = start + local_n;

    // Reservar memoria solo para las filas necesarias + 2 filas frontera
    float** A    = malloc((local_n + 2) * sizeof(float*));
    float** Anew = malloc((local_n + 2) * sizeof(float*));
    for (int i = 0; i < local_n + 2; i++) {
        A[i] = calloc(m, sizeof(float));
        Anew[i] = calloc(m, sizeof(float));
    }

    // Inicializar condiciones de frontera
    float* y = malloc(n * sizeof(float));
    for (int i = 0; i < n; i++)
        y[i] = sinf(pi * i / (n - 1));

    for (int i = 1; i <= local_n; i++) {
        int global_i = start + i - 1;
        A[i][0] = y[global_i];
        A[i][m - 1] = y[global_i] * expf(-pi);
    }
    if (rank == 0) {
        for (int j = 0; j < m; j++) A[1][j] = 0.0f; // frontera superior
    }
    if (rank == size - 1) {
        for (int j = 0; j < m; j++) A[local_n][j] = 0.0f; // frontera inferior
    }

    free(y);

    double start_time = MPI_Wtime();

    while (error > tol && iter < iter_max) {
        MPI_Request reqs[4];
        // Comunicación no bloqueante de filas frontera
        if (rank > 0)
            MPI_Irecv(A[0], m, MPI_FLOAT, rank - 1, 0, MPI_COMM_WORLD, &reqs[0]);
        if (rank < size - 1)
            MPI_Irecv(A[local_n + 1], m, MPI_FLOAT, rank + 1, 1, MPI_COMM_WORLD, &reqs[1]);
        if (rank > 0)
            MPI_Isend(A[1], m, MPI_FLOAT, rank - 1, 1, MPI_COMM_WORLD, &reqs[2]);
        if (rank < size - 1)
            MPI_Isend(A[local_n], m, MPI_FLOAT, rank + 1, 0, MPI_COMM_WORLD, &reqs[3]);

        // Calcular valores internos (sin depender de frontera aún)
        for (int i = 2; i < local_n; i++) {
            for (int j = 1; j < m - 1; j++) {
                Anew[i][j] = 0.25f * (A[i][j + 1] + A[i][j - 1] + A[i - 1][j] + A[i + 1][j]);
            }
        }

        // Esperar frontera
        if (rank > 0) MPI_Wait(&reqs[0], MPI_STATUS_IGNORE);
        if (rank < size - 1) MPI_Wait(&reqs[1], MPI_STATUS_IGNORE);
        if (rank > 0) MPI_Wait(&reqs[2], MPI_STATUS_IGNORE);
        if (rank < size - 1) MPI_Wait(&reqs[3], MPI_STATUS_IGNORE);

        // Calcular bordes dependientes de la comunicación
        for (int j = 1; j < m - 1; j++) {
            Anew[1][j] = 0.25f * (A[1][j + 1] + A[1][j - 1] + A[0][j] + A[2][j]);
            Anew[local_n][j] = 0.25f * (A[local_n][j + 1] + A[local_n][j - 1] + A[local_n - 1][j] + A[local_n + 1][j]);
        }

        float local_error = 0.0f;
        for (int i = 1; i <= local_n; i++)
            for (int j = 1; j < m - 1; j++)
                local_error = fmaxf(local_error, sqrtf(fabsf(Anew[i][j] - A[i][j])));

        for (int i = 1; i <= local_n; i++)
            for (int j = 1; j < m - 1; j++)
                A[i][j] = Anew[i][j];

        MPI_Allreduce(&local_error, &error, 1, MPI_FLOAT, MPI_MAX, MPI_COMM_WORLD);

        iter++;
        if (rank == 0 && iter % (iter_max / 10) == 0)
            printf("%5d, %0.6f\n", iter, error);
    }

    double end_time = MPI_Wtime();
    if (rank == 0)
        printf("Laplace MPI took %f seconds\n", end_time - start_time);

    for (int i = 0; i < local_n + 2; i++) {
        free(A[i]);
        free(Anew[i]);
    }
    free(A);
    free(Anew);

    MPI_Finalize();
    return 0;
}
