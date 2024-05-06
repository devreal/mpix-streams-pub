#include <iostream>
#include <mpi.h>
#include "streams.h"
#include "greq.h"
#ifdef MPIX_STREAMS_HAVE_HIP
#include <hip/hip_runtime.h>
#endif // MPIX_STREAMS_HAVE_HIP

namespace mpix::streams {

typedef struct recv_args : public greq_stat_t {
  void *buf;
  int count;
  MPI_Datatype dt;
  int dest;
  int tag;
  MPI_Comm comm;
} recv_args_t;

static int greq_query_cb(void *data, MPI_Status *status);

static int greq_free_cb(void *extra_state);

static int greq_cancel_cb(void *extra_state, int complete);

static int recv_complete_cb(int err, void *data) {
  /* complete the GREQ */
  recv_args_t *args = (recv_args_t*)data;
  return PMPI_Grequest_complete(args->greq);
}

static void recv_start_cb(void *data) {
  /* thread-shift to progress thread */
  recv_args_t *args = (recv_args_t*)data;
  MPI_Request req;
  /* start the operation */
  //std::cout << "IRECV: calling MPI_Irecv" << std::endl;
  PMPI_Irecv(args->buf, args->count, args->dt, args->dest, args->tag, args->comm, &req);
  /* attach a callback that will complete the GREQ */
  MPIX_Continue(&req, &recv_complete_cb, data, MPIX_CONT_REQBUF_VOLATILE, &args->stat, mpix::streams::cont_req);
}

static void enqueue_recv_start_cb(void *data) {
  /* thread-shift to progress thread */
  op_queue.push([=](){
    recv_start_cb(data);
  });
}

} // mpix::streams


int MPI_Irecv(
  void *buf, int count,
  MPI_Datatype dt, int dest,
  int tag, MPI_Comm comm,
  MPI_Request* req)
{
  int ret;
  namespace MS = mpix::streams;
  MS::stream_attr_val_t *attr = MS::get_comm_stream(comm);

  if (NULL == attr) {
    /* no stream attached, pass through */
    return PMPI_Irecv(buf, count, dt, dest, tag, comm, req);
  }

#ifdef MPIX_STREAMS_HAVE_HIP
  if (std::holds_alternative<hipStream_t>(attr->stream)) {
    MS::recv_args_t *args = new MS::recv_args_t;
    args->buf = buf;
    args->count = count;
    args->dt = dt;
    args->dest = dest;
    args->tag = tag;
    args->comm = comm;
    PMPI_Grequest_start(&MS::greq_query_cb<MS::recv_args_t>, &MS::greq_free_cb<MS::recv_args_t>,
                        &MS::greq_cancel_cb<MS::recv_args_t>, args, req);
    args->greq = *req;
    MS::rmap.add(*req, attr);
    hipStream_t hip_stream = std::get<hipStream_t>(attr->stream);
    MS::stream_traits trait{hip_stream};
    if (!MS::use_events() || trait.is_capturing()) {
      hipLaunchHostFunc(hip_stream, &MS::enqueue_recv_start_cb, args);
    } else {
      MS::events.add_event(hip_stream, [=](){ MS::recv_start_cb(args); });
    }
  } else
#endif // MPIX_STREAMS_HAVE_HIP
  if (std::holds_alternative<std::monostate>(attr->stream)) {
    ret = PMPI_Irecv(buf, count, dt, dest, tag, comm, req);
  }

  return ret;
}
