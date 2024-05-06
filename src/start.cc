#include <iostream>
#include <mpi.h>
#include <cassert>

#include "streams.h"

#ifdef MPIX_STREAMS_HAVE_HIP
#include <hip/hip_runtime.h>
#endif // MPIX_STREAMS_HAVE_HIP

namespace mpix::streams {

  static void op_start_cb(void* data) {
    MPI_Request *req = (MPI_Request*)data;
    /* start the operation */
    //std::cout << "START: calling MPI_Start" << std::endl;
    MPI_Request tmp = *req;
    PMPI_Start(req);
    assert(*req == tmp); // don't support changing the greq
  }

  static void enqueue_op_start_cb(void *data) {
    //std::cout << "START: enqueuing start to op_queue" << std::endl;
    /* thread-shift to progress thread */
    op_queue.push([=](){
      op_start_cb(data);
    });
  }

} // namespace mpix::streams

int MPI_Start(MPI_Request *req)
{
  int ret = MPI_SUCCESS;
  namespace MS = mpix::streams;
  MS::stream_attr_val_t *attr;
  attr = MS::rmap.get(*req);

  if (NULL == attr) {
    /* no stream attached, pass through */
    return PMPI_Start(req);
  }

#ifdef MPIX_STREAMS_HAVE_HIP
  if (std::holds_alternative<hipStream_t>(attr->stream)) {
    hipStream_t hip_stream = std::get<hipStream_t>(attr->stream);
    MS::stream_traits trait{hip_stream};
    if (!MS::use_events() || trait.is_capturing()) {
      hipLaunchHostFunc(hip_stream, &MS::enqueue_op_start_cb, req);
    } else {
      MS::events.add_event(hip_stream, [=](){ MS::op_start_cb(req); });
    }
    return ret;
  } else
#endif // MPIX_STREAMS_HAVE_HIP
  if (std::holds_alternative<std::monostate>(attr->stream)) {
    ret = PMPI_Start(req);
  }

  return ret;
}