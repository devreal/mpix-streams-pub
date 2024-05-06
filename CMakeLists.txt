cmake_minimum_required(VERSION 3.22)
project(mpix-streams VERSION 0.1 LANGUAGES C CXX DESCRIPTION "Implementation of the MPIX_Streams API")

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
include(CheckIncludeFile)
include(CheckSymbolExists)
include(CheckLanguage)

find_package(MPI REQUIRED)
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

# check if we have MPI continuations
set(CMAKE_REQUIRED_LIBRARIES MPI::MPI_C)
check_include_file("mpi-ext.h" MPI_HAVE_MPIEXTH)
if (MPI_HAVE_MPIEXTH)
  check_symbol_exists(MPIX_Continue "mpi-ext.h" MPI_HAVE_CONTINUE)
elseif(MPI_HAVE_MPIEXTH)
  message(STATUS "Looking for MPI Continuations without mpi-ext.h")
  check_symbol_exists(MPIX_Continue "mpi-ext.h" MPI_HAVE_CONTINUE)
endif(MPI_HAVE_MPIEXTH)

if (NOT MPI_HAVE_CONTINUE)
  message(FATAL_ERROR "MPI implementation with the MPI Continuation extension is required!")
endif(NOT MPI_HAVE_CONTINUE)

# find HIP
set(TTG_HIP_PLATFORM "__HIP_PLATFORM_AMD__" CACHE STRING "Which platform to use when compiling HIP-related code (default: __HIP_PLATFORM_AMD__)")

# HIP LANGUAGE introduced in 3.21
check_language(HIP)
if(CMAKE_HIP_COMPILER)
  enable_language(HIP)
  find_package(hip)
endif(CMAKE_HIP_COMPILER)
set(MPIX_STREAMS_HAVE_HIP ${CMAKE_HIP_COMPILER} CACHE BOOL "True if support for HIP has been detected")


set(MPIX_STREAM_SOURCES src/comm_init.cc
                        src/comm_sync_stream.cc
                        src/init.cc
                        src/irecv.cc
                        src/isend.cc
                        src/recv_init.cc
                        src/send_init.cc
                        src/start.cc
                        src/stream_set.cc
                        src/stream_wait.cc)

add_library(mpix-streams SHARED ${MPIX_STREAM_SOURCES})
add_library(MPIXStreams::mpix-streams ALIAS mpix-streams)
target_link_libraries(mpix-streams PUBLIC Threads::Threads MPI::MPI_C $<$<BOOL:${MPIX_STREAMS_HAVE_HIP}>:hip::host>)
set_target_properties(mpix-streams PROPERTIES VERSION ${PROJECT_VERSION})
set_target_properties(mpix-streams PROPERTIES PUBLIC_HEADER include/mpix_streams.h)
set_target_properties(mpix-streams PROPERTIES INSTALL_RPATH ${CMAKE_INSTALL_LIBDIR})
set_target_properties(mpix-streams PROPERTIES CXX_STANDARD 17)
target_compile_definitions(mpix-streams PRIVATE $<$<BOOL:${MPIX_STREAMS_HAVE_HIP}>:MPIX_STREAMS_HAVE_HIP=1>)


# Configure the installation paths
set(MPIX_STREAMS_INSTALL_INCLUDEDIR ${CMAKE_INSTALL_INCLUDEDIR})
set(MPIX_STREAMS_INSTALL_LIBDIR ${CMAKE_INSTALL_LIBDIR})
set(MPIX_STREAMS_INSTALL_BINDIR ${CMAKE_INSTALL_BINDIR})
set(MPIX_STREAMS_INSTALL_LIBEXECDIR ${CMAKE_INSTALL_LIBEXECDIR})
set(MPIX_STREAMS_INSTALL_DATADIR ${CMAKE_INSTALL_DATADIR})
set(MPIX_STREAMS_INSTALL_CMAKEDIR ${CMAKE_INSTALL_DATADIR}/cmake/mpix-streams)

# check if we're being built as a sub-project
STRING(COMPARE EQUAL ${CMAKE_CURRENT_BINARY_DIR} ${CMAKE_CURRENT_SOURCE_DIR} MPIX_STREAMS_BUILD_INPLACE)
if(NOT MPIX_STREAMS_BUILD_INPLACE)
  include_directories(BEFORE "${CMAKE_CURRENT_SOURCE_DIR}/include")
endif(NOT MPIX_STREAMS_BUILD_INPLACE)
include_directories(BEFORE "${CMAKE_CURRENT_BINARY_DIR}")

# make sure the include directory is part of the interface
target_include_directories(mpix-streams
        INTERFACE
        $<INSTALL_INTERFACE:${MPIX_STREAMS_INSTALL_INCLUDEDIR}>
        $<$<NOT:${MPIX_STREAMS_BUILD_INPLACE}>:
          $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}>>
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>)

install(TARGETS mpix-streams
        EXPORT mpix-streams
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(EXPORT mpix-streams # rules to export the targets built here
        FILE MPIXStreamsTargets.cmake
        NAMESPACE MPIXStreams::
        DESTINATION ${MPIX_STREAMS_INSTALL_CMAKEDIR})

configure_package_config_file(mpixstreams-config.cmake.in
                              ${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/mpixstreams-config.cmake
                              INSTALL_DESTINATION ${MPIX_STREAMS_INSTALL_CMAKEDIR}
                              PATH_VARS   MPIX_STREAMS_INSTALL_LIBDIR
                                          MPIX_STREAMS_INSTALL_INCLUDEDIR
                                          MPIX_STREAMS_INSTALL_BINDIR
                                          MPIX_STREAMS_INSTALL_LIBEXECDIR
                                          MPIX_STREAMS_INSTALL_CMAKEDIR)

install(FILES ${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/mpixstreams-config.cmake
        DESTINATION ${MPIX_STREAMS_INSTALL_CMAKEDIR})

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/mpix-streams.pc.in ${CMAKE_CURRENT_BINARY_DIR}/mpix-streams.pc @ONLY)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/mpix-streams.pc
        DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/pkgconfig)

install(FILES)


add_subdirectory(benchmarks)
add_subdirectory(tests)
