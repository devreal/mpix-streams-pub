

if (MPIX_STREAMS_HAVE_HIP)
  add_executable(latency_overlap latency_overlap.hip)
  target_link_libraries(latency_overlap PUBLIC MPIXStreams::mpix-streams MPI::MPI_C)
  target_include_directories(latency_overlap PRIVATE ${MPI_C_INCLUDE_DIRS})

  add_executable(compute_overlap compute_overlap.hip compute/compute_kernel.hip)
  target_link_libraries(compute_overlap PUBLIC MPIXStreams::mpix-streams MPI::MPI_C)
  target_include_directories(compute_overlap PRIVATE ${MPI_C_INCLUDE_DIRS})
endif(MPIX_STREAMS_HAVE_HIP)
