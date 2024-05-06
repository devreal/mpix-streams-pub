#include <mpi.h>
#include <array>
#include <atomic>
#include <functional>
#include <iostream>
#include <map>
#include <mutex>
#include <queue>
#include <stack>
#include <variant>
#include <stdexcept>
#include <pthread.h>
#include <mpi-ext.h>
#ifdef MPIX_STREAMS_HAVE_HIP
#include <hip/hip_runtime.h>
#endif // MPIX_STREAMS_HAVE_HIP

namespace mpix::streams {

inline int comm_keyval;

inline MPI_Request cont_req;

// TODO: add more types here
using stream_t = std::variant<
                  std::monostate // empty
#ifdef MPIX_STREAMS_HAVE_HIP
                , hipStream_t
#endif // MPIX_STREAMS_HAVE_HIP
                  >;

// TODO: add more types here
using event_t = std::variant<
                  std::monostate // empty
#ifdef MPIX_STREAMS_HAVE_HIP
                , hipEvent_t
#endif // MPIX_STREAMS_HAVE_HIP
                  >;

// fwd-decl
struct stream_attr_val_t;

// info passed to completion of wait
struct wait_info_t {
  stream_attr_val_t *attr;
  uint64_t sigval;
  MPI_Request req; // used for persistent requests
  MPI_Status* stat; // used for persistent requests

  wait_info_t(stream_attr_val_t *attr,
              uint64_t sigval,
              MPI_Request req = MPI_REQUEST_NULL,
              MPI_Status *stat = MPI_STATUS_IGNORE)
  : attr(attr)
  , sigval(sigval)
  , req(req)
  , stat(stat)
  { }

  friend bool operator<(const wait_info_t& l, const wait_info_t& r)
  {
      return l.sigval < r.sigval;
  }
};

inline bool operator> (const wait_info_t& lhs, const wait_info_t& rhs) { return rhs < lhs; }
inline bool operator<=(const wait_info_t& lhs, const wait_info_t& rhs) { return !(lhs > rhs); }
inline bool operator>=(const wait_info_t& lhs, const wait_info_t& rhs) { return !(lhs < rhs); }

using queue_type = std::priority_queue<wait_info_t, std::vector<wait_info_t>, std::greater<wait_info_t>>;

// comm attribute data
struct stream_attr_val_t {
  uint64_t *signal_mem = nullptr;
  stream_t signal_stream;
  stream_t stream;
  std::atomic<uint64_t> signal_cnt = 0;
  uint64_t complete_sig = 0;
  std::mutex wait_mtx;
  queue_type queue;
  std::vector<std::unique_ptr<wait_info_t>> waitinfo; // graph wait infos
  bool was_capturing = false; // flips to true if this stream was used to capture a graph

  stream_attr_val_t(stream_t stream)
  : stream(stream)
  { }

  void add_waitinfo(std::unique_ptr<wait_info_t> info) {
    std::lock_guard g{wait_mtx};
    waitinfo.push_back(std::move(info));
  }
};

/* to track assignment of requests to streams */
struct request_stream_map {
private:
  mutable std::mutex req_map_mtx;
  std::map<MPI_Request, stream_attr_val_t*> req_map;

public:

  void add(const MPI_Request& req, stream_attr_val_t* attr) {
    std::lock_guard g{req_map_mtx};
    req_map.insert({req, attr});
  }

  void remove(const MPI_Request& req) {
    std::lock_guard g{req_map_mtx};
    req_map.erase(req);
  }

  stream_attr_val_t* get(const MPI_Request& req) const {
    std::lock_guard g{req_map_mtx};
    auto it = req_map.find(req);
    if (it == req_map.end()) {
      return nullptr;
    } else {
      return it->second;
    }
  }
};

inline request_stream_map rmap;

using start_cb_t = std::function<void()>;

/**
 * A thread-safe queue to store operation start callbacks.
 * We use std::function for type punning.
 */
struct op_start_queue {
private:
  mutable std::mutex mtx;
  std::queue<start_cb_t> queue;

public:
  void push(start_cb_t cb) {
    std::lock_guard g{mtx};
    queue.push(std::move(cb));
  }

  void start_one() {
    start_cb_t cb;
    if (!queue.empty())
    {
      {
        std::lock_guard g{mtx};
        if (!queue.empty()) {
          cb = std::move(queue.front());
          queue.pop();
        }
      }
      // invoke the callback if we have one
      if (cb) {
        cb();
      }
    }
  }

  void start_all() {
    constexpr const std::size_t at_once = 16;
    std::array<start_cb_t, at_once> cb_arr;
    std::size_t pos = 0;
    while (!queue.empty())
    {
      { // extract at_once callbacks and release the lock
        pos = 0;
        std::lock_guard g{mtx};
        for (; pos < at_once; ++pos) {
          if (!queue.empty()) {
            cb_arr[pos] = std::move(queue.front());
            queue.pop();
          } else {
            break;
          }
        }
      }
      // invoke the callbacks if we have one
      for (std::size_t i = 0; i < pos; ++i) {
        cb_arr[i]();
      }
    }
  }
};

inline op_start_queue op_queue;

int stream_comm_initialize();

int stream_comm_finalize();

static inline stream_attr_val_t *get_comm_stream(MPI_Comm comm)
{
  int ret, flag;
  stream_attr_val_t *attr = NULL;
  ret = MPI_Comm_get_attr(comm, comm_keyval, &attr, &flag);
  if (MPI_SUCCESS != ret) {
      return NULL;
  }
  return attr;
}

template<typename Stream = void>
struct stream_traits
{
  template<typename T>
  static bool is_capturing(T&& dummy) {
    throw std::runtime_error("unsupprted stream type");
  }
};


/* overload for HIP */
#ifdef MPIX_STREAMS_HAVE_HIP
template<>
struct stream_traits<hipStream_t>
{
  hipStream_t m_stream;
  stream_traits(hipStream_t stream)
  : m_stream(stream)
  { }

  bool is_capturing() {
    hipStreamCaptureStatus capture;
    hipError_t herr;
    herr = hipStreamIsCapturing(m_stream, &capture);
    return (capture == hipStreamCaptureStatusActive);
  }
};
#endif // MPIX_STREAMS_HAVE_HIP

template<typename T>
stream_traits(T) -> stream_traits<T>;

void delete_stream_attr(stream_attr_val_t *attr);

inline std::atomic<int> poll_enabled;
void poll_init();

struct event_pool {

  using callback_t = std::function<void()>;

private:
  struct event_cb {
    event_t event;
    callback_t cb;
  };

  // pool of idle events
  std::stack<event_t>   m_idle_pool;
  std::vector<event_cb> m_active_pool;
  std::mutex m_mtx;

public:
  template<typename Callback>
  void add_event(stream_t stream, Callback&& cb) {
    std::lock_guard g{m_mtx};
    event_cb ecb;
    if (!m_idle_pool.empty()) {
      // use an allocated event
      ecb.event = m_idle_pool.top();
      m_idle_pool.pop();
    } else {
#ifdef MPIX_STREAMS_HAVE_HIP
      hipEvent_t event;
      if (hipSuccess != hipEventCreateWithFlags(&event, hipEventDisableTiming)) {
        std::cout << "WARN: hipEventCreateWithFlags failed!" << std::endl;
      }
      ecb.event = event;
#endif // MPIX_STREAMS_HAVE_HIP
    }
    ecb.cb = std::forward<Callback>(cb);
#ifdef MPIX_STREAMS_HAVE_HIP
    if (hipSuccess != hipEventRecord(std::get<hipEvent_t>(ecb.event), std::get<hipStream_t>(stream))) {
      std::cout << "WARN: hipEventRecord failed!" << std::endl;
    }
#endif // MPIX_STREAMS_HAVE_HIP
    m_active_pool.push_back(ecb);
  }

  void progress_all() {
    if (m_active_pool.size() > 0) {
      std::vector<event_cb> tmp_ecbs;
      std::size_t complete = 0;
      {
        std::lock_guard g{m_mtx};
        std::swap(tmp_ecbs, m_active_pool);
      }
      for (auto it = tmp_ecbs.begin(); it != tmp_ecbs.end() - complete; /* increment inline */) {
        bool done = true;
#ifdef MPIX_STREAMS_HAVE_HIP
        done = (hipSuccess == hipEventQuery(std::get<hipEvent_t>(it->event)));
#endif // MPIX_STREAMS_HAVE_HIP
        if (done) {
          // event complete, invoke the callback
          it->cb();
          // move to end of vector for easier removal
          if (it !=  tmp_ecbs.end()-(complete+1)) {
            std::swap(*it, *(tmp_ecbs.end()-(complete+1)));
          }
          complete++;
          // keep iterator where it is, we just moved in a new element
        } else {
          it++;
        }
      }

      // clean up temporaries and completed events
      {
        std::lock_guard g{m_mtx};
        // put complete events into idle list
        for (auto it = tmp_ecbs.end()-complete; it != tmp_ecbs.end(); ++it) {
          m_idle_pool.push(it->event);
        }
        // resize to remove complete events
        tmp_ecbs.resize(tmp_ecbs.size() - complete);
        if (0 < tmp_ecbs.size()) {
          // move incomplete events back
          if (m_active_pool.size() == 0) {
            // easy: just move back
            std::swap(tmp_ecbs, m_active_pool);
          } else {
            // need to merge: insert old elements at the end
            m_active_pool.insert(m_active_pool.end(), tmp_ecbs.begin(), tmp_ecbs.end());
          }
        }
      }
    }
  }
};

inline event_pool events;

inline bool use_events() {
  static std::once_flag oflag;
  static bool use_events_ = true;
  std::call_once(oflag, [&](){
    std::string envstr;
    const char *envstr_c = std::getenv("MPIX_STREAM_USE_EVENTS");
    if (envstr_c) {
      envstr = envstr_c;
    }
    if (envstr == "false") {
      use_events_ = false;
    } else {
      try {
        if (std::stoi(envstr) == 0) {
          use_events_ = false;
        }
      } catch (std::invalid_argument&)
      { }
    }
  });
  return use_events_;
}

} // namespace mpix::streams
