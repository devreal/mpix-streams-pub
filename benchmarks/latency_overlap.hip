#include <iostream>
#include <algorithm>
#include <mpi.h>
#include <mpix_streams.h>
#include <hip/hip_runtime.h>

#define MAXSIZE (1024*1024)
#define BLOCKSIZE (size_t)256
#define GRIDSIZE(_size, _blocksize) (((_size)+(_blocksize)-1)/(_blocksize))
#define NUM_REPS 10000

__global__ void fill(/*hipLaunchParm lp,*/ unsigned *buf, size_t N)
{
    unsigned gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < N) {
        buf[gid] = gid;
    }
}


int main(int argc, char **argv)
{
    int rank, size, flag;
    MPI_Request req;
    MPI_Comm stream_comm;
    unsigned *buf;
    int *res_dev = NULL, *res_host;
    hipStream_t stream = 0;
    std::size_t blockSize = BLOCKSIZE;
    bool overlap = true;

    if (argc > 1) {
        overlap = false;
    }

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

    /**
     * Start stream region
     */
    hipMalloc((void**)&buf, MAXSIZE*sizeof(buf[0]));
    for (size_t count = 1; count <= 1024*1024; count *= 2) {
        double start, end;
        double sum = 0.0;
        for (int i = 0; i < NUM_REPS; ++i) {
            start = MPI_Wtime();
            if (rank == 0) {
                fill<<<dim3(GRIDSIZE(count, blockSize)), dim3(std::min(count, blockSize)), 0, stream>>>(buf, count);
                if (overlap) {
                    MPI_Isend(buf, count, MPI_FLOAT, 1, 0, stream_comm, &req);
                    MPIX_Stream_wait(&req, MPI_STATUS_IGNORE);
                    MPIX_Comm_sync_stream(stream_comm);
                } else {
                    hipStreamSynchronize(stream);
                    MPI_Send(buf, count, MPI_FLOAT, 1, 0, MPI_COMM_WORLD);
                }
            } else if (rank == 1) {
                if (overlap) {
                    MPI_Irecv(buf, count, MPI_FLOAT, 0, 0, stream_comm, &req);
                    MPIX_Stream_wait(&req, MPI_STATUS_IGNORE);
                    MPIX_Comm_sync_stream(stream_comm);
                } else {
                    MPI_Recv(buf, count, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                }
            }
            end = MPI_Wtime();
            sum += end - start;
        }
        if (rank == 0) {
            printf("fill-send %zu B %.3f us\n", count, sum/NUM_REPS/1E-6);
        }
    }
    hipFree(buf);
    MPI_Comm_free(&stream_comm);
    MPI_Finalize();
    return 0;
}
