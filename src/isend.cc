#include <iostream>
#include <mpi.h>
#include "streams.h"
#include "greq.h"
#ifdef MPIX_STREAMS_HAVE_HIP
#include <hip/hip_runtime.h>
#endif // MPIX_STREAMS_HAVE_HIP

namespace mpix::streams {

typedef struct send_args : public greq_stat_t {
  const void *buf;
  int count;
  MPI_Datatype dt;
  int dest;
  int tag;
  MPI_Comm comm;
} send_args_t;

static int send_complete_cb(int err, void *data) {
  /* complete the GREQ */
  send_args_t *args = (send_args_t*)data;
  //std::cout << "ISEND: isend complete, completing greq" << std::endl;
  return PMPI_Grequest_complete(args->greq);
}

static void send_start_cb(void *data) {
  send_args_t *args = (send_args_t*)data;
  MPI_Request req;
  /* start the operation */
  //std::cout << "ISEND: calling MPI_Isend" << std::endl;
  PMPI_Isend(args->buf, args->count, args->dt, args->dest, args->tag, args->comm, &req);
  /* attach a callback that will complete the GREQ */
  MPIX_Continue(&req, &send_complete_cb, data, MPIX_CONT_REQBUF_VOLATILE, &args->stat, mpix::streams::cont_req);
}

static void enqueue_send_start_cb(void *data) {
  //std::cout << "ISEND: enqueuing isend to op_queue" << std::endl;
  /* thread-shift to progress thread */
  op_queue.push([=](){
    send_start_cb(data);
  });
}

} // namespace mpix::streams


int MPI_Isend(
  const void *buf, int count,
  MPI_Datatype dt, int dest,
  int tag, MPI_Comm comm,
  MPI_Request* req)
{
  int ret;
  namespace MS = mpix::streams;
  MS::stream_attr_val_t *attr = MS::get_comm_stream(comm);

  if (NULL == attr) {
    /* no stream attached, pass through */
    return PMPI_Isend(buf, count, dt, dest, tag, comm, req);
  }

#ifdef MPIX_STREAMS_HAVE_HIP
  if (std::holds_alternative<hipStream_t>(attr->stream)) {
    MS::send_args_t *args = new MS::send_args_t;
    args->buf = buf;
    args->count = count;
    args->dt = dt;
    args->dest = dest;
    args->tag = tag;
    args->comm = comm;
    PMPI_Grequest_start(&MS::greq_query_cb<MS::send_args_t>, &MS::greq_free_cb<MS::send_args_t>,
                        &MS::greq_cancel_cb<MS::send_args_t>, args, req);
    args->greq = *req;
    MS::rmap.add(*req, attr);
    hipStream_t hip_stream = std::get<hipStream_t>(attr->stream);
    MS::stream_traits trait{hip_stream};
    if (!MS::use_events() || trait.is_capturing()) {
      hipLaunchHostFunc(hip_stream, &MS::enqueue_send_start_cb, args);
    } else {
      MS::events.add_event(hip_stream, [=](){ MS::send_start_cb(args); });
    }
  }
#else  // MPIX_STREAMS_HAVE_HIP
  if (std::holds_alternative<std::monostate>(attr->stream)) {
    ret = PMPI_Isend(buf, count, dt, dest, tag, comm, req);
  }
#endif // MPIX_STREAMS_HAVE_HIP

  return ret;
}
