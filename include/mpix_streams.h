#ifndef MPIX_STREAMS_H
#define MPIX_STREAMS_H

#include <mpi.h>

#if defined(c_plusplus) || defined(__cplusplus)
extern "C" {
#endif

/**
 * Associate a stream of specified kind with comm.
 * The stream_ptr should point a variable holding the stream
 * of the specified kind. The variable is not required to
 * remain accessible after a call to this procedure returns.
 * If the implementation supports the specified stream kind,
 * flag will be set to 1 and 0 otherwise.
 * If flag is 1 upon return, operations issued on comm
 * will be ordered with operations enqeued on the provided stream
 * prior to this call.
 * The kind of the stream must match one of the memory kind names outlined
 * in the MPI side document "Memory Allocation Kinds".
 * No restrictors are allowed to be contained in the kind specification.
 */
int MPIX_Comm_set_stream(MPI_Comm comm, const char *kind, void *stream_ptr,
                         MPI_Info info, int *flag);

/**
 * Remove any stream currently associated with comm.
 * Subsequent operations issued on comm will behave as if
 * there was never a stream associated with comm.
 */
int MPIX_Comm_delete_stream(MPI_Comm comm);

/**
 * Block the stream associated with comm until the operation req
 * has completed. The status is filled before the stream is released.
 * MPI_STATUS_IGNORE may be passed instead.*/
int MPIX_Stream_wait(MPI_Request *req, MPI_Status *stat);

/**
 * Block the stream associated with comm until the operations reqs
 * have completed. The statuses are filled before the execution continues.
 * MPI_STATUSES_IGNORE may be passed instead.
 */
int MPIX_Stream_waitall(int count, MPI_Request reqs[], MPI_Status stats[]);

/**
 * Synchronize the stream attached to the communicator to complete.
 */
int MPIX_Comm_sync_stream(MPI_Comm comm);


#if defined(c_plusplus) || defined(__cplusplus)
}
#endif

#endif // MPIX_STREAMS_H
