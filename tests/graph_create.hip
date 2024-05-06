/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil -*- */
#include <iostream>
#include <vector>
#include <mpi.h>
#include <mpix_streams.h>
#include <hip/hip_runtime.h>

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


void start_func (void* userData)
{
  MPI_Request *req = (MPI_Request *)userData;

  MPI_Start (req);
}

void stream_waitall_func (void* userData)
{
    void **userbuf = (void**)userData;
    MPI_Request *reqs = (MPI_Request*)userbuf[0];
    int count = *(int*)userbuf[1];

    MPIX_Stream_waitall(count, reqs, MPI_STATUS_IGNORE);
}


#define NSTEPS 10

int main(int argc, char **argv)
{
    int rank, size, flag, partner;
    MPI_Request req[2];
    int *buf_dev, *tmp_dev;
    const unsigned blockSize = BLOCKSIZE;
    size_t SIZE = 128;
    int i=0;
    int nreqs=2;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    partner = (rank+1)%size;

    hipStream_t stream;
    HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
    MPIX_Comm_set_stream(MPI_COMM_WORLD, "hip", &stream, MPI_INFO_NULL, &flag);
    if (!flag) {
      std::cout << "MPIX_Comm_set_stream failed to set the stream!" << std::endl;
      std::abort();
    }

    HIP_CHECK(hipMalloc((void**)&buf_dev, SIZE*sizeof(int)));
    HIP_CHECK(hipMalloc((void**)&tmp_dev, SIZE*sizeof(int)));

    MPI_Recv_init(tmp_dev, SIZE, MPI_INT, partner, 0, MPI_COMM_WORLD, &req[0]);
    MPI_Send_init(buf_dev, SIZE, MPI_INT, partner, 0, MPI_COMM_WORLD, &req[1]);

    bool graphCaptured = false;
    hipGraph_t graph;
    hipGraphExec_t instance;

    HIP_CHECK(hipGraphCreate(&graph, 0));

    hipGraphNode_t kernelNode1;
    hipKernelNodeParams kernelNodeParams1 = {0};
    kernelNodeParams1.func = (void*)zero;
    kernelNodeParams1.gridDim = dim3(SIZE/blockSize);
    kernelNodeParams1.blockDim = dim3(blockSize);
    kernelNodeParams1.sharedMemBytes = 0;
    void* kernel1Args[2] = { (void*)tmp_dev, (void*)&SIZE};
    kernelNodeParams1.kernelParams = kernel1Args;
    kernelNodeParams1.extra = NULL;
    HIP_CHECK(hipGraphAddKernelNode(&kernelNode1, graph, NULL, 0, &kernelNodeParams1));

    hipGraphNode_t hostNode1;
    hipHostNodeParams hostNodeParams1 = {0};
    hostNodeParams1.fn = start_func;
    void *host1Params[1] = { (void*)&req[0]};
    hostNodeParams1.userData = host1Params;
    HIP_CHECK(hipGraphAddHostNode(&hostNode1, graph, &kernelNode1, 1, &hostNodeParams1));

    hipGraphNode_t kernelNode2;
    hipKernelNodeParams kernelNodeParams2 = {0};
    kernelNodeParams2.func = (void*)zero;
    kernelNodeParams2.gridDim = dim3(SIZE/blockSize);
    kernelNodeParams2.blockDim = dim3(blockSize);
    kernelNodeParams2.sharedMemBytes = 0;
    void* kernel2Args[3] = { (void*)tmp_dev, (void*)&SIZE, (void*)&i};
    kernelNodeParams2.kernelParams = kernel1Args;
    kernelNodeParams2.extra = NULL;
    HIP_CHECK(hipGraphAddKernelNode(&kernelNode2, graph, &kernelNode1, 1, &kernelNodeParams2));

    hipGraphNode_t hostNode2;
    hipHostNodeParams hostNodeParams2 = {0};
    hostNodeParams2.fn = start_func;
    void *host2Params[1] = {&req[1]};
    hostNodeParams2.userData = host2Params;
    HIP_CHECK(hipGraphAddHostNode(&hostNode2, graph, &kernelNode2, 1, &hostNodeParams2));

    hipGraphNode_t hostNode3;
    hipHostNodeParams hostNodeParams3 = {0};
    hostNodeParams3.fn = stream_waitall_func;
    void *host3Params[2] = {(void*)req, (void*)&nreqs};
    hostNodeParams3.userData = host3Params;
    std::vector<hipGraphNode_t> nodeDependencies;
    nodeDependencies.push_back(hostNode1);
    nodeDependencies.push_back(hostNode2);
    HIP_CHECK(hipGraphAddHostNode(&hostNode3, graph, nodeDependencies.data(), nodeDependencies.size(),
                                  &hostNodeParams3));

    hipGraphNode_t kernelNode3;
    hipKernelNodeParams kernelNodeParams3 = {0};
    kernelNodeParams3.func = (void*)check;
    kernelNodeParams3.gridDim = dim3(SIZE/blockSize);
    kernelNodeParams3.blockDim = dim3(blockSize);
    kernelNodeParams3.sharedMemBytes = 0;
    void* kernel3Args[3] = { (void*)tmp_dev, (void*)&SIZE, (void*)&i};
    kernelNodeParams3.kernelParams = kernel1Args;
    kernelNodeParams3.extra = NULL;
    HIP_CHECK(hipGraphAddKernelNode(&kernelNode3, graph, &hostNode3, 1, &kernelNodeParams3));

    HIP_CHECK(hipGraphInstantiate(&instance, graph, NULL, NULL, 0));

    for (i=0; i<NSTEPS; i++)  {
        std::cout << rank << ": Iteration " << i << " launching graph" << std::endl;
        HIP_CHECK(hipGraphLaunch(instance, stream));
        MPIX_Comm_sync_stream(MPI_COMM_WORLD);
    }

    HIP_CHECK(hipGraphExecDestroy(instance));
    HIP_CHECK(hipGraphDestroy(graph));
    HIP_CHECK(hipStreamDestroy(stream));
    HIP_CHECK(hipFree(buf_dev));
    HIP_CHECK(hipFree(tmp_dev));

    MPI_Finalize();
    return 0;
}
