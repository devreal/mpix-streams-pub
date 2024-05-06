#ifndef MPIX_STREAMS_GREQ_H
#define MPIX_STREAMS_GREQ_H

#include <mpi.h>

namespace mpix::streams {

struct greq_stat_t {
  MPI_Request greq;
  MPI_Status stat;
};

template<typename Args>
inline int greq_query_cb(void *data, MPI_Status *status)
{
  Args *args = (Args*)data;
  *status = args->stat;
  return MPI_SUCCESS;
}

template<typename Args>
inline int greq_free_cb(void *data)
{
  Args *args = (Args*)data;
  /* free the recv arguments */
  delete args;
  return MPI_SUCCESS;
}

template<typename Args>
inline int greq_cancel_cb(void *data, int complete)
{
  /* nothing to do */
  return MPI_SUCCESS;
}


} // namespace mpix::streams

#endif // MPIX_STREAMS_GREQ_H