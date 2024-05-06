#include <iostream>
#include <mpi.h>
#include <mpix_streams.h>
#include <hip/hip_runtime.h>

#define SIZE (1024*1024)
#define BLOCKSIZE 256
#define GRIDSIZE(_size, _blocksize) (((_size)+(_blocksize)-1)/(_blocksize))

__global__ void fill(/*hipLaunchParm lp,*/ unsigned *buf, size_t N)
{
    unsigned gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < N) {
        buf[gid] = gid;
    }
}

__global__ void check(/*hipLaunchParm lp,*/ unsigned *buf, int *res, size_t N)
{
    unsigned gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < N) {
        res[gid] = (buf[gid] == gid) ? 1 : 0;
    }
}



int main(int argc, char **argv)
{
    int rank, size, flag;
    MPI_Request req;
    unsigned *buf;
    int *res_dev = NULL, *res_host;
    hipStream_t stream = 0;
    const unsigned blockSize = BLOCKSIZE;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    hipStreamCreateWithFlags(&stream, hipStreamNonBlocking);

    MPIX_Comm_set_stream(MPI_COMM_WORLD, "hip", &stream, MPI_INFO_NULL, &flag);

    if (!flag) {
        std::cout << "MPIX_Comm_set_stream failed to set the stream!" << std::endl;
        std::abort();
    }

    /**
     * Start stream region
     */
    hipMallocAsync((void**)&buf, SIZE*sizeof(buf[0]), stream);
    if (rank == 0) {
        fill<<<dim3(GRIDSIZE(SIZE, blockSize)), dim3(blockSize), 0, stream>>>(buf, SIZE);
        MPI_Isend(buf, SIZE, MPI_FLOAT, 1, 0, MPI_COMM_WORLD, &req);
        MPIX_Stream_wait(&req, MPI_STATUS_IGNORE);
    } else if (rank == 1) {
        MPI_Irecv(buf, SIZE, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, &req);
        hipMallocAsync((void**)&res_dev, SIZE*sizeof(res_dev[0]), stream);
        MPIX_Stream_wait(&req, MPI_STATUS_IGNORE);
        res_host = (int*)malloc(SIZE*sizeof(res_host[0]));
        check<<<dim3(GRIDSIZE(SIZE, blockSize)), dim3(blockSize), 0, stream>>>(buf, res_dev, SIZE);
        hipMemcpyAsync(res_host, res_dev, SIZE*sizeof(res_host[0]), hipMemcpyDeviceToHost, stream);
    }
    hipFreeAsync(buf, stream);
    MPIX_Comm_sync_stream(MPI_COMM_WORLD);

    /**
     * End stream region
     */


    if (rank == 1) {
        for (int i = 0; i < SIZE; ++i) {
            if (res_host[i] != 1) {
                std::cerr << "Result in pos " << i << ": " << res_host[i] << std::endl;
            }
        }
        free(res_host);
    }

    MPI_Finalize();
    return 0;
}
