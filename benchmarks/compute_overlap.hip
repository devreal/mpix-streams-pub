#include <iostream>
#include <algorithm>
#include <chrono>

#include <mpi.h>
#include <mpix_streams.h>
#include <hip/hip_runtime.h>

#include "compute/compute_kernel.h"

#define MAXSIZE (1024*1024)
#define BLOCKSIZE (size_t)256
#define GRIDSIZE(_size, _blocksize) (((_size)+(_blocksize)-1)/(_blocksize))
#define NUM_REPS 10
#define WARMUP 5

int main(int argc, char **argv)
{
    int rank, size;
    float *buf;
    int *res_dev = NULL, *res_host;
    hipStream_t stream = 0;
    std::size_t blockSize = BLOCKSIZE;
    bool overlap = true;

    if (argc > 1) {
        overlap = false;
    }

    MPI_Request req;
    MPI_Comm stream_comm;
    int flag;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    hipStreamCreateWithFlags(&stream, hipStreamNonBlocking);
    MPI_Comm_dup(MPI_COMM_WORLD, &stream_comm);
    MPIX_Comm_set_stream(stream_comm, "hip", &stream, MPI_INFO_NULL, &flag);

    if (!flag) {
        std::cout << "MPIX_Comm_set_stream failed to set the stream!" << std::endl;
        std::abort();
    }
    compute_params_t compute_params;
    compute_init(compute_params, stream);

    /**
     * Start stream region
     */
    hipMalloc((void**)&buf, compute_params.N*sizeof(double));
    assert(compute_params.N > 64*1024*1024);
    /* dummy setup */
    compute_set_params(compute_params, 0.0001);
    for (int niter = 1; niter <= 1000; niter *= 10) {
        compute_params.niter = niter;
        compute_params.K = 0;
        for (int count = 1024; count <= 64*1024*1024; count *= 2) {
            double start, end;
            double sum = 0.0;
            //compute_params.N = count;
            for (int i = 0; i < NUM_REPS+WARMUP; ++i) {
                if (i == WARMUP) {
                    /* reset after warmup */
                    sum = 0.0;
                }
                auto start = std::chrono::high_resolution_clock::now();
                if (rank == 0) {
                    compute_launch(compute_params);
                    if (overlap) {
                        MPI_Isend(compute_params.Afdevice, count, MPI_DOUBLE, 1, 0, stream_comm, &req);
                        compute_launch(compute_params);
                        MPIX_Stream_wait(&req, MPI_STATUS_IGNORE);
                        MPIX_Comm_sync_stream(stream_comm);
                    } else {
                        hipStreamSynchronize(stream);
                        //MPI_Send(compute_params.Afdevice, count, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD);
                        MPI_Isend(compute_params.Afdevice, count, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD, &req);
                        compute_launch(compute_params);
                        MPI_Wait(&req, MPI_STATUS_IGNORE);
                        hipStreamSynchronize(stream);
                    }
                } else if (rank == 1) {
                    compute_launch(compute_params);
                    if (overlap) {
                        MPI_Irecv(buf, count, MPI_DOUBLE, 0, 0, stream_comm, &req);
                        compute_launch(compute_params);
                        MPIX_Stream_wait(&req, MPI_STATUS_IGNORE);
                        MPIX_Comm_sync_stream(stream_comm);
                    } else {
                        hipStreamSynchronize(stream);
                        //MPI_Recv(buf, count, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                        MPI_Irecv(buf, count, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &req);
                        compute_launch(compute_params);
                        MPI_Wait(&req, MPI_STATUS_IGNORE);
                        hipStreamSynchronize(stream);
                    }
                }
                auto end = std::chrono::high_resolution_clock::now();
                sum += std::chrono::duration<double>(end-start).count();
            }
            if (rank == 0) {
                double avg = sum/NUM_REPS;
                printf("compute-send-compute N %zu niter %d K %d total %.3f us \n",
                        count*sizeof(double), niter, compute_params.K, avg/1E-6);

            }
        }
    }
    compute_fini(compute_params);
    hipFree(buf);
    MPI_Comm_free(&stream_comm);
    MPI_Finalize();
    return 0;
}
