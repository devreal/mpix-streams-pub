#include <iostream>
#include <mpi.h>
#include "streams.h"


int MPI_Recv_init(
  void *buf, int count,
  MPI_Datatype dt, int src,
  int tag, MPI_Comm comm,
  MPI_Request* req)
{
  int ret;
  namespace MS = mpix::streams;
  MS::stream_attr_val_t *attr = MS::get_comm_stream(comm);
  ret = PMPI_Recv_init(buf, count, dt, src, tag, comm, req);

  if (NULL != attr) {
    /* no stream attached, pass through */
    MS::rmap.add(*req, attr);
  }

  return ret;
}
