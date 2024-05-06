
#include <mpi.h>
#include<cstdlib>
#include "streams.h"
#include "mpix_streams.h"

#ifdef MPIX_STREAMS_HAVE_HIP
#include <hip/hip_runtime.h>
#endif // MPIX_STREAMS_HAVE_HIP

namespace mpix::streams {
  void delete_stream_attr(stream_attr_val_t *attr) {
    if (NULL != attr) {
      /* clean up the attribute */
#ifdef MPIX_STREAMS_HAVE_HIP
      hipFree(attr->signal_mem);
      hipStreamDestroy(std::get<hipStream_t>(attr->signal_stream));
      attr->signal_stream = std::monostate{};
#endif // MPIX_STREAMS_HAVE_HIP
      delete attr;
      attr = NULL;
    }
  }
}

static bool need_progress(MPI_Info info)
{
  bool res = false;
  std::string progress_env;
  const char *env_str = std::getenv("MPIX_STREAM_PROGRESS");
  if (env_str != nullptr) {
    progress_env = env_str;
  }

  /* lambda to check the string */
  auto check_progress_str = [](std::string& str) {
    int value = 0;
    try {
      value = std::stoi(str);
    } catch (std::invalid_argument&)
    { }
    if (value || str == "yes") {
      return true;
    }
    return false;
  };

  if (check_progress_str(progress_env)) {
    res = true;
  } else if (info != MPI_INFO_NULL) {
    char buf[128];
    int len = 128;
    int flag;
    MPI_Info_get_string(info, "mpix_stream_progress", &len, buf, &flag);
    if (flag) {
      progress_env = buf;
      res = check_progress_str(progress_env);
    }
  }
  return res;
}

int MPIX_Comm_set_stream(MPI_Comm comm, const char* kind, void *stream_ptr, MPI_Info info, int *flag)
{
  int ret;
  void *signal_mem;
  *flag = 0;

  namespace MS = mpix::streams;
  mpix::streams::stream_attr_val_t *attr = mpix::streams::get_comm_stream(comm);
  if (!MS::poll_enabled) {
    if (need_progress(info)) {
      MS::poll_init();
    }
  }
#ifdef MPIX_STREAMS_HAVE_HIP
  if (0 == strncmp(kind, "hip", 3)) {
    attr = new mpix::streams::stream_attr_val_t{*(hipStream_t*)stream_ptr};
    hipExtMallocWithFlags((void**)&attr->signal_mem, sizeof(int64_t), hipMallocSignalMemory);
    hipStreamWriteValue64(*(hipStream_t*)stream_ptr, attr->signal_mem, 0, 0);
    hipStream_t hip_stream;
    hipStreamCreateWithFlags(&hip_stream, hipStreamNonBlocking);
    attr->signal_stream = hip_stream;
    *flag = 1;
  } else
#endif // MPIX_STREAMS_HAVE_HIP
  {
    attr->stream = std::monostate{};
  }
  ret = MPI_Comm_set_attr(comm, mpix::streams::comm_keyval, attr);
  return ret;
}

int MPIX_Comm_delete_stream(MPI_Comm comm)
{
  mpix::streams::delete_stream_attr(mpix::streams::get_comm_stream(comm));
  return MPI_SUCCESS;
}
