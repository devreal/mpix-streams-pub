
#include <stdio.h>
#include <stdlib.h>

#include <hip/hip_runtime_api.h>
#include <hip/hip_runtime.h>

#include <random>
#include <chrono>

#include "compute_kernel.h"

__global__ void compute_me(long *A, double *F, int N, int K, int niter)
{
    for (int k=0; k<niter; k++ ) {
        for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x) {
            long val     = A[i];
            long temp    = val;
            for (int k = 0; k < K; k++) {
                temp += val;
            }
            double fval  = F[i];
            double ftemp = fval;
            for (int k = 0; k < K; k++) {
                ftemp += fval;
            }
            A[i] = temp;
            F[i] = ftemp;
        }
    }
}

static void init_buf(long *array, double *farray,  int dim)
{
    std::minstd_rand generator;
    std::uniform_int_distribution<> distribution(1, 128);
    for ( int i = 0; i < dim; ++i) {
        array[i]  = distribution(generator);
        farray[i] = (double) distribution(generator);
    }
}

int compute_init (compute_params_t &params, hipStream_t stream)
{
    //Hardcoding these parameters for now, can revisit later if necessary.
    params.N       = 64*1024*1024;
    params.K       = 13604;
    params.Kthresh = 110;
    params.Rthresh = 2;

    params.Ahost  = (long*) malloc (params.N * sizeof(long));
    params.Afhost = (double*) malloc (params.N * sizeof(double));
    if (NULL == params.Ahost || NULL == params.Afhost) {
        return 1;
    }
    init_buf(params.Ahost, params.Afhost, params.N);

    HIP_CHECK(hipMalloc((void**)&params.Adevice, params.N*sizeof(long)));
    HIP_CHECK(hipMemcpy(params.Adevice, params.Ahost, params.N*sizeof(long), hipMemcpyDefault));
    HIP_CHECK(hipMalloc((void**)&params.Afdevice, params.N*sizeof(double)));
    HIP_CHECK(hipMemcpy(params.Afdevice, params.Afhost, params.N*sizeof(double), hipMemcpyDefault));
    params.stream = stream;

    hipDeviceProp_t prop;
    int deviceId;

    HIP_CHECK(hipGetDevice(&deviceId));
    HIP_CHECK(hipGetDeviceProperties(&prop, deviceId));
    params.threadsPerBlock = 256;
    if (prop.maxThreadsPerBlock > 0) {
        params.threadsPerBlock = prop.maxThreadsPerBlock;
    }

    return 0;
}

void compute_set_params(compute_params_t &params, double runtime)
{
    double t1;
    int prev_K=0;
    do {
        auto t1s = std::chrono::high_resolution_clock::now();
        params.niter = 1;
        compute_launch (params);
        HIP_CHECK(hipStreamSynchronize(params.stream));
        auto t1e = std::chrono::high_resolution_clock::now();
        t1 = std::chrono::duration<double>(t1e-t1s).count();

        if (t1 > runtime* params.Rthresh) {
            params.K /= 2;
        }
        prev_K = params.K;
    } while (params.K > params.Kthresh && prev_K != params.K);

    auto t10s = std::chrono::high_resolution_clock::now();
    params.niter = 10;
    compute_launch (params);
    HIP_CHECK(hipStreamSynchronize(params.stream));
    auto t10e = std::chrono::high_resolution_clock::now();
    double t10 = std::chrono::duration<double>(t10e-t10s).count();

    double slope = (t10 - t1)/9.0;
    double dist  = t10 - slope * 10;

    double est = (runtime - dist)/slope;
    long estimated_niter = std::lround(est);
    params.niter = estimated_niter < 1 ? 1: (int)estimated_niter;

    auto ts = std::chrono::high_resolution_clock::now();
    compute_launch (params);
    HIP_CHECK(hipStreamSynchronize(params.stream));
    auto te = std::chrono::high_resolution_clock::now();

    params.est_runtime = std::chrono::duration<double>(te-ts).count();
    //printf("runtime: %lf estimated niter %d K %d actual runtime %lf\n", runtime, params.niter, params.K, params.est_runtime);
    return;
}

void compute_launch (compute_params_t &params)
{
    compute_me<<<dim3(params.N/params.threadsPerBlock),
                 dim3(params.threadsPerBlock),
                 0, params.stream>>>(params.Adevice,
                                     params.Afdevice,
                                     params.N,
                                     params.K,
                                     params.niter);
}

void compute_fini(compute_params_t &params)
{
    // Not sure we need these next two lines
    HIP_CHECK(hipMemcpy(params.Ahost, params.Adevice, params.N*sizeof(long), hipMemcpyDefault));
    HIP_CHECK(hipMemcpy(params.Afhost, params.Afdevice, params.N*sizeof(double), hipMemcpyDefault));

    HIP_CHECK(hipFree(params.Adevice));
    HIP_CHECK(hipFree(params.Afdevice));
    free (params.Ahost);
    free (params.Afhost);
}
