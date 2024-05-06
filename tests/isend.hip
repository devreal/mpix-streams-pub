#include <iostream>
#include <mpi.h>
#include <mpix_streams.h>
#include <hip/hip_runtime.h>

#define SIZE (128)
#define BLOCKSIZE 16
#define GRIDSIZE(_size, _blocksize) (((_size)+(_blocksize)-1)/(_blocksize))

__global__ void fill(/*hipLaunchParm lp,*/ unsigned *buf, size_t N)
{
    unsigned gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < N) {
        buf[gid] = gid;
    }
}

__global__ void zero(/*hipLaunchParm lp,*/ unsigned *buf, size_t N)
{
    unsigned gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < N) {
        buf[gid] = 0;
    }
}


int main(int argc, char **argv)
{
    int rank, size, flag;
    MPI_Request req, recv_req;
    unsigned *buf_dev, *buf_host = NULL;
    hipStream_t stream = hipStreamDefault;
    const unsigned blockSize = BLOCKSIZE;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int recv_rank = (rank+1)%size;

    hipStreamCreateWithFlags(&stream, hipStreamNonBlocking);
    if (rank == 0) {
        MPIX_Comm_set_stream(MPI_COMM_WORLD, "hip", &stream, MPI_INFO_NULL, &flag);
    }

    for (int i = 0; i < 10; ++i) {
        /* post a recv */
        buf_host = (unsigned*)calloc(SIZE, sizeof(buf_host[0]));
        if (rank == recv_rank) {
            MPI_Irecv(buf_host, SIZE, MPI_INT, 0, 0, MPI_COMM_WORLD, &recv_req);
        }

        /**
         * Send stream region
         */
        if (rank == 0) {
            hipMallocAsync((void**)&buf_dev, SIZE*sizeof(buf_dev[0]), stream);
            fill<<<dim3(SIZE/blockSize), dim3(blockSize), 0, stream>>>(buf_dev, SIZE);
            if (!flag) {
                std::cout << "MPIX_Comm_set_stream failed to set the stream!" << std::endl;
                std::abort();
            }
            MPI_Isend(buf_dev, SIZE, MPI_INT, recv_rank, 0, MPI_COMM_WORLD, &req);
            MPIX_Stream_wait(&req, MPI_STATUS_IGNORE);
            hipFreeAsync(buf_dev, stream);
            MPIX_Comm_sync_stream(MPI_COMM_WORLD);
        }

        /* wait for recv */
        if (rank == recv_rank) {
            MPI_Wait(&recv_req, MPI_STATUS_IGNORE);
            for (unsigned i = 0; i < SIZE; ++i) {
                if (buf_host[i] != i) {
                    std::cout << "Expected " << i << " but got " << buf_host[i] << std::endl;
                }
            }
            free(buf_host);
            buf_host = NULL;
        }
    }

    MPI_Finalize();
    return 0;
}
