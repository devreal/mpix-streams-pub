#include <iostream>
#include <mpi.h>
#include "mpix_streams.h"
#include "streams.h"

namespace mpix::streams {
  static int wait_complete_cb(int err, void *data);
  static void wait_stream_cb(void *data);
} // namespace mpix::streams

int MPIX_Stream_wait(MPI_Request *req, MPI_Status *stat)
{
  int ret;
  namespace MS = mpix::streams;
  mpix::streams::stream_attr_val_t *attr;
  attr = mpix::streams::rmap.get(*req);
  if (NULL != attr) {
    /* block the stream until we release it */
#ifdef MPIX_STREAMS_HAVE_HIP
    if (std::holds_alternative<hipStream_t>(attr->stream)) {
      int pi;
      hipError_t ret;
      hipDeviceGetAttribute(&pi, hipDeviceAttributeCanUseStreamWaitValue, 0);
      if (!pi) std::cout << "WARN: streamWaitValue not supported on device!" << std::endl;
      hipStream_t hip_stream = std::get<hipStream_t>(attr->stream);
      MS::stream_traits trait{hip_stream};
      if (!attr->was_capturing && trait.is_capturing()) {
        /* TODO: should this be reset at some point? */
        attr->was_capturing = true;
        /* first time recording -> add a node to reset the signal memory */
        hipStreamWriteValue64(hip_stream, attr->signal_mem, 0, 0);
      }

      uint64_t sigval = ++attr->signal_cnt; // TODO: relax memory ordering

      /* block the stream */
      //std::cout << "SWAIT: stream " << hip_stream << " waiting for value " << sigval << std::endl;
      ret = hipStreamWaitValue64(hip_stream, attr->signal_mem, sigval, hipStreamWaitValueGte, (uint64_t)-1);
      //ret = hipStreamWaitValue32(hip_stream, attr->signal_mem, sigval, hipStreamWaitValueGte, (uint32_t)-1);
      if (ret != hipSuccess) std::cout << "WARN: hipStreamWaitValue64 failed!" << std::endl;

      if (trait.is_capturing()) {
        /* enqueue a host function that registers the callback to release the stream */
        std::unique_ptr<mpix::streams::wait_info_t> info = std::make_unique<mpix::streams::wait_info_t>(attr, sigval, *req, stat);
        hipLaunchHostFunc(hip_stream, &mpix::streams::wait_stream_cb, info.get());
        /* enqueue the unique ptr so that it is destroyed when we delete the stream */
        attr->add_waitinfo(std::move(info));
        /* TODO: reset to NULL if it's a non-persistent request (if we ever allow that for capturing) */
      } else {
        /* register the callback directly */
        mpix::streams::wait_info_t *info = new mpix::streams::wait_info_t{attr, sigval};
        MPIX_Continue(req, &mpix::streams::wait_complete_cb, info, MPIX_CONT_REQBUF_VOLATILE, stat, mpix::streams::cont_req);
      }
    } else
#endif // MPIX_STREAMS_HAVE_HIP
    {
      ret = MPI_Wait(req, stat);
    }
  } else {
    ret = MPI_Wait(req, stat);
  }

  return ret;
}


namespace mpix::streams {
static int wait_complete_cb(int err, void *data)
{
  wait_info_t *info = (wait_info_t *)data;
  stream_attr_val_t *attr = info->attr;
  std::lock_guard g{attr->wait_mtx};
  if (info->sigval > attr->complete_sig + 1) {
    //std::cout << "SWAIT: stream " << std::get<hipStream_t>(attr->stream) << " delaying release (" << info->sigval << " > " << attr->complete_sig << "+1)" << std::endl;
    /* we are not the next to release so defer */
    attr->queue.push(*info);
    delete info;
  } else {
    /* we are next, release this and all pending after us */
    attr->complete_sig++;
    while (!attr->queue.empty()) {
      const wait_info_t& next = attr->queue.top();
      if (next.sigval == attr->complete_sig + 1) {
        /* direct successor, release */
        attr->queue.pop();
        attr->complete_sig++;
      } else {
        /* found a gap, stop releasing */
        break;
      }
    }
    /* set the signal variable */
#ifdef MPIX_STREAMS_HAVE_HIP
    hipStream_t hip_stream = std::get<hipStream_t>(attr->signal_stream);
    hipError_t ret;
    //std::cout << "SWAIT: stream " << hip_stream << " writing value " << attr->complete_sig << std::endl;
    ret = hipStreamWriteValue64(hip_stream, attr->signal_mem, attr->complete_sig, 0);
    if (ret != hipSuccess) std::cout << "WARN: hipStreamWaitValue64 failed!" << std::endl;
#endif // MPIX_STREAMS_HAVE_HIP
  }
  return MPI_SUCCESS;
}

void wait_stream_cb(void *data){
  mpix::streams::wait_info_t *info = static_cast<mpix::streams::wait_info_t*>(data);
  stream_attr_val_t *attr = info->attr;
  if (attr->was_capturing) {
    /* reset the complete_sig counter if we find that the graph is executed again */
    if (attr->complete_sig > info->sigval) {
      attr->complete_sig = 0;
    }
  }
  assert(info->req != MPI_REQUEST_NULL);
  MPIX_Continue(&info->req, &mpix::streams::wait_complete_cb, info, MPIX_CONT_REQBUF_VOLATILE, info->stat, mpix::streams::cont_req);
}

} // namespace mpix::streams
