
#ifndef COMPUTE_KERNEL_H
#define COMPUTE_KERNEL_H

#include <mpi.h>
#include <hip/hip_runtime_api.h>

#define HIP_CHECK(condition) {                                            \
        hipError_t error = condition;                                     \
        if(error != hipSuccess){                                          \
            fprintf(stderr,"HIP error: %d line: %d\n", error,  __LINE__); \
            MPI_Abort(MPI_COMM_WORLD, error);                             \
        }                                                                 \
    }


typedef struct compute_params_s {
    int         N, K, Kthresh, Rthresh, niter;
    int         threadsPerBlock;
    long       *Ahost, *Adevice;
    double     *Afhost, *Afdevice;
    double      est_runtime;
    hipStream_t stream;
    bool        free_stream;
} compute_params_t;

int  compute_init(compute_params_t &params, hipStream_t stream);
void compute_set_params(compute_params_t &params, double runtime);
void compute_launch (compute_params_t &params);
void compute_fini(compute_params_t &params);

#endif // COMPUTE_KERNEL_H
