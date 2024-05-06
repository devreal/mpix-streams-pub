#include <mpi.h>
#include <time.h>
#include "streams.h"

namespace mpix::streams {

static pthread_t poll_thread;

static
int delete_attr_cb(MPI_Comm comm, int comm_keyval,
                   void *attribute_val, void *extra_state);

void *thread_poll_main(void *ptr) {
  struct timespec ts;
  ts.tv_sec = 0;
  ts.tv_nsec = 1000; // sleep one microsecond
  while (poll_enabled.load(std::memory_order_relaxed)) {
    int flag;
    /* start available ops */
    op_queue.start_all();
    if (use_events()) {
      /* check events */
      events.progress_all();
    }
    /* check for completes ops */
    PMPI_Test(&cont_req, &flag, MPI_STATUS_IGNORE);
    if (flag) {
      PMPI_Start(&cont_req);
    }
    nanosleep(&ts, NULL); // release the CPU
  }
  return NULL;
}

int stream_comm_initialize() {
  static bool initialized = false;

  if (initialized) return MPI_SUCCESS;

  static pthread_mutex_t init_mtx = PTHREAD_MUTEX_INITIALIZER;

  pthread_mutex_lock(&init_mtx);
  if (!initialized) {
      /* create a keyval
        * TODO: add extra state and handle it in the callbacks
        */
      int ret;
      ret = MPI_Comm_create_keyval(MPI_COMM_NULL_COPY_FN, &delete_attr_cb,
                                   &comm_keyval, NULL);
      if (MPI_SUCCESS != ret) {
          return ret;
      }
      MPIX_Continue_init(0, 0, MPI_INFO_NULL, &cont_req);
      initialized = true;
  }
  pthread_mutex_unlock(&init_mtx);
  return MPI_SUCCESS;
}

void poll_init()
{
  if (0 == std::atomic_exchange(&poll_enabled, 1)) {
    pthread_create(&poll_thread, NULL, &thread_poll_main, NULL);
  }
}

int stream_comm_finalize() {
  if (poll_enabled) {
    poll_enabled = 0; // stop polling thread
    pthread_join(poll_thread, NULL); // wait for polling thread to complete
  }
  return PMPI_Comm_free_keyval(&mpix::streams::comm_keyval);
}

static
int delete_attr_cb(MPI_Comm comm, int comm_keyval,
                   void *attribute_val, void *extra_state)
{
    stream_attr_val_t *attr = (stream_attr_val_t*)attribute_val;
    if (NULL != attr){
      delete_stream_attr(attr);
    }
    return MPI_SUCCESS;
}

} // namespace mpix::streams