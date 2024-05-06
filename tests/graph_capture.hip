/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil -*- */
#include <iostream>
#include <mpi.h>
#include <mpix_streams.h>
#include <hip/hip_runtime.h>

#define SIZE (128)
#define BLOCKSIZE 16
#define GRIDSIZE(_size, _blocksize) (((_size)+(_blocksize)-1)/(_blocksize))

#define HIP_CHECK(condition) {                                            \
        hipError_t error = condition;                                     \
        if(error != hipSuccess){                                          \
            fprintf(stderr,"HIP error: %d line: %d\n", error,  __LINE__); \
            MPI_Abort(MPI_COMM_WORLD, error);                             \
        }                                                                 \
    }


__global__ void fill(int *buf, size_t N, int offset)
{
    unsigned gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < N) {
        buf[gid] = gid + offset;
    }
}

__global__ void zero(int *buf, size_t N)
{
    unsigned gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < N) {
        buf[gid] = 0;
    }
}

__global__ void check(int *buf, size_t N, int offset)
{
    unsigned gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < N) {
        if (buf[gid] == gid + offset) {
            // be happy
        }
        else {
            // be unhappy, not returning error right now
        }
    }
}

#define NSTEPS 10

int main(int argc, char **argv)
{
    int rank, size, flag, partner;
    MPI_Request req[2];
    int *buf_dev, *tmp_dev;
    const unsigned blockSize = BLOCKSIZE;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    partner = (rank+1)%size;

    hipStream_t stream;
    HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
    hipStream_t stream2;
    HIP_CHECK(hipStreamCreateWithFlags(&stream2, hipStreamNonBlocking));
    MPIX_Comm_set_stream(MPI_COMM_WORLD, "hip", &stream, MPI_INFO_NULL, &flag);
    if (!flag) {
      std::cout << "MPIX_Comm_set_stream failed to set the stream!" << std::endl;
      std::abort();
    }

    HIP_CHECK(hipMalloc((void**)&buf_dev, SIZE*sizeof(int)));
    HIP_CHECK(hipMalloc((void**)&tmp_dev, SIZE*sizeof(int)));

#ifdef USE_PERSISTENT_OPS
    MPI_Recv_init(tmp_dev, SIZE, MPI_INT, partner, 0, MPI_COMM_WORLD, &req[0]);
    MPI_Send_init(buf_dev, SIZE, MPI_INT, partner, 0, MPI_COMM_WORLD, &req[1]);
#endif

    bool graphCaptured = false;
    hipGraph_t graph;
    hipGraphExec_t instance;
    for (int i=0; i<NSTEPS; i++)  {
        if (!graphCaptured) {
            std::cout << rank << ": Iteration " << i << " starting StreamCapture" << std::endl;
            HIP_CHECK(hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal));
            zero<<<dim3(SIZE/blockSize), dim3(blockSize), 0, stream>>>(tmp_dev, SIZE);
#ifdef USE_PERSISTENT_OPS
            MPI_Start(&req[0]);
#else
            MPI_Irecv(tmp_dev, SIZE, MPI_INT, partner, 0, MPI_COMM_WORLD, &req[0]);
#endif
            fill<<<dim3(SIZE/blockSize), dim3(blockSize), 0, stream>>>(buf_dev, SIZE, i);
#ifdef USE_PERSISTENT_OPS
            MPI_Start(&req[1]);
#else
            MPI_Isend(buf_dev, SIZE, MPI_INT, partner, 0, MPI_COMM_WORLD, &req[1]);
#endif

#ifdef HAVE_STREAM_WAITALL
            MPIX_Stream_waitall(2, req, MPI_STATUS_IGNORE);
#else
            MPIX_Stream_wait(&req[0], MPI_STATUS_IGNORE);
            MPIX_Stream_wait(&req[1], MPI_STATUS_IGNORE);
#endif

            check<<<dim3(SIZE/blockSize), dim3(blockSize), 0, stream>>>(tmp_dev, SIZE, i);
            HIP_CHECK(hipStreamEndCapture(stream, &graph));
            std::cout << rank << ": Iteration " << i << " ending StreamCapture" << std::endl;
            HIP_CHECK(hipGraphInstantiate(&instance, graph, NULL, NULL, 0));
            graphCaptured = true;
            /* change the associated stream so that we can use sync-stream below */
            MPIX_Comm_set_stream(MPI_COMM_WORLD, "hip", &stream2, MPI_INFO_NULL, &flag);
        }
        else {
            std::cout << rank << ": Iteration " << i << " launching graph" << std::endl;
            HIP_CHECK(hipGraphLaunch(instance, stream2));
            MPIX_Comm_sync_stream(MPI_COMM_WORLD);
        }
    }


    HIP_CHECK(hipFree(buf_dev));
    HIP_CHECK(hipFree(tmp_dev));
    HIP_CHECK(hipStreamDestroy(stream));

    MPI_Finalize();
    return 0;
}
