#include <mpi.h>
#include "streams.h"
#include "mpix_streams.h"


int MPIX_Comm_sync_stream(MPI_Comm comm)
{
  int rc = MPI_SUCCESS;
  namespace MS = mpix::streams;
  MS::stream_attr_val_t *attr = MS::get_comm_stream(comm);

  if (attr != nullptr) {
    while (rc == MPI_SUCCESS) {
      int flag;
      /* check events */
      if (MS::use_events()) {
        MS::events.progress_all();
      }

      /* progress stream */
#ifdef MPIX_STREAMS_HAVE_HIP
      if (std::holds_alternative<hipStream_t>(attr->stream)) {
        hipStream_t hip_stream = std::get<hipStream_t>(attr->stream);
        hipError_t herr = hipStreamQuery(hip_stream);
        if (herr == hipSuccess) {
          break;
        } else if (herr != hipErrorNotReady) {
          rc = MPI_ERR_INTERN;
          break;
        }
      } else
#endif // MPIX_STREAMS_HAVE_HIP
      {
        /* if we don't support the stream we bail out right away */
        rc = MPI_ERR_INTERN;
        break;
      }
      if (!MS::poll_enabled) {
        /* start available ops */
        MS::op_queue.start_all();
        /* test continuation request */
        rc = MPI_Test(&MS::cont_req, &flag, MPI_STATUS_IGNORE);
        if (flag) {
          rc = MPI_Start(&MS::cont_req);
        }
      }
    }
  }

  return rc;
}

