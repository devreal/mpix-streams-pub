#include <iostream>
#include <mpi.h>
#include "streams.h"

/**
 * Intercept MPI_Init, MPI_Init_thread, and MPI_Finalize
 * to initialize and tear-down internal state.
 */

int MPI_Init(int *argc, char ***argv)
{
  int provided, ret;
  ret = PMPI_Init_thread(argc, argv, MPI_THREAD_MULTIPLE, &provided);
  if (provided != MPI_THREAD_MULTIPLE) {
    std::cerr << "MPI library does not provide MPI_THREAD_MULTIPLE" << std::endl;
    MPI_Abort(MPI_COMM_WORLD, MPI_ERR_INTERN);
  }
  return mpix::streams::stream_comm_initialize();
}


int MPI_Init_thread(int *argc, char ***argv,
                    int requested, int *provided)
{
  PMPI_Init_thread(argc, argv, MPI_THREAD_MULTIPLE, provided);
  if (*provided != MPI_THREAD_MULTIPLE) {
    std::cerr << "MPI library does not provide MPI_THREAD_MULTIPLE" << std::endl;
    MPI_Abort(MPI_COMM_WORLD, MPI_ERR_INTERN);
  }
  return mpix::streams::stream_comm_initialize();
}

int MPI_Finalize()
{
  mpix::streams::stream_comm_finalize();
  return PMPI_Finalize();
}
