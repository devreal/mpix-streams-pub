#include <iostream>
#include <mpi.h>
#include "streams.h"


int MPI_Send_init(
  const void *buf, int count,
  MPI_Datatype dt, int dest,
  int tag, MPI_Comm comm,
  MPI_Request* req)
{
  int ret;
  namespace MS = mpix::streams;
  MS::stream_attr_val_t *attr = MS::get_comm_stream(comm);
  ret = PMPI_Send_init(buf, count, dt, dest, tag, comm, req);

  if (NULL != attr) {
    /* link attributes to request */
    MS::rmap.add(*req, attr);
  }

  return ret;
}
