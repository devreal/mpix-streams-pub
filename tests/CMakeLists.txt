

if (MPIX_STREAMS_HAVE_HIP)
  add_executable(isendrecv_hip isend_irecv.hip)
  target_link_libraries(isendrecv_hip PUBLIC MPIXStreams::mpix-streams MPI::MPI_C)
  target_include_directories(isendrecv_hip PRIVATE ${MPI_C_INCLUDE_DIRS})


  add_executable(isend_hip isend.hip)
  target_link_libraries(isend_hip PUBLIC MPIXStreams::mpix-streams MPI::MPI_C)
  target_include_directories(isend_hip PRIVATE ${MPI_C_INCLUDE_DIRS})

  add_executable(graph_capture_hip graph_capture.hip)
  target_link_libraries(graph_capture_hip PUBLIC MPIXStreams::mpix-streams MPI::MPI_C)
  target_include_directories(graph_capture_hip PRIVATE ${MPI_C_INCLUDE_DIRS})
  target_compile_definitions(graph_capture_hip PUBLIC USE_PERSISTENT_OPS)

endif(MPIX_STREAMS_HAVE_HIP)
