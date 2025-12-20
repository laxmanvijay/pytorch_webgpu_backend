cmake_minimum_required(VERSION 3.14)
project(gloo-headers-only LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(FetchContent)

set(TORCH_TAG "v2.5.0")
set(GLOO_TAG "5354032ea08eadd7fc4456477f7f7c6308818509")

# Declare Gloo to be fetched
FetchContent_Declare(
  gloo
  GIT_REPOSITORY https://github.com/facebookincubator/gloo.git
  GIT_TAG ${GLOO_TAG}
)

# Populate Gloo without building it
FetchContent_GetProperties(gloo)
if(NOT gloo_POPULATED)
  FetchContent_Populate(gloo)
endif()

# Gloo's CMakeLists are not written such that we can fetch its CMake variables
# without building it (with FetchContent_MakeAvailable, add_subdirectory etc..).
# Therefore we have to manually set the variables we care about, i.e. those used
# in Gloo's in config.h.in
#
# We must set these variables in the same way that Torch is setting them:
# https://github.com/pytorch/pytorch/blob/v2.5.0/third_party/gloo.BUILD
set(GLOO_VERSION_MAJOR 0)
set(GLOO_VERSION_MINOR 5)
set(GLOO_VERSION_PATCH 0)
set(GLOO_USE_CUDA 1)
set(GLOO_USE_NCCL 0)
set(GLOO_USE_ROCM 0)
set(GLOO_USE_RCCL 0)
set(GLOO_USE_REDIS 0)
set(GLOO_USE_MPI 0)
set(GLOO_USE_AVX 0)
set(GLOO_USE_LIBUV 0)
set(GLOO_HAVE_TRANSPORT_TCP_TLS 1)
set(GLOO_HAVE_TRANSPORT_TCP 0)
set(GLOO_HAVE_TRANSPORT_IBVERBS 0)
set(GLOO_HAVE_TRANSPORT_UV 0)

# Now we can create config.h
# Configure config.h.in into config.h
configure_file(
  ${gloo_SOURCE_DIR}/gloo/config.h.in
  ${CMAKE_CURRENT_BINARY_DIR}/gloo-headers/gloo/config.h
  # @ONLY  # Use @ONLY if you only want to substitute @VAR@, not #cmakedefine
)

# At this point we are good and we could just include gloo_SOURCE_DIR (plus config.h),
# but this will also make available the .cpp files. Instead we can gather all .h
# files, same way Torch does, and copy them in a separate dir
# https://github.com/pytorch/pytorch/blob/32f585d9346e316e554c8d9bf7548af9f62141fc/third_party/gloo.BUILD#L32

set(GLOO_HEADER_PATTERNS
  "gloo/*.h"
  "gloo/common/*.h"
  "gloo/rendezvous/*.h"
  "gloo/transport/*.h"
  "gloo/transport/tcp/*.h"
  "gloo/transport/tcp/tls/*.h"
)

set(GLOO_HEADERS "")
foreach(pattern ${GLOO_HEADER_PATTERNS})
    file(GLOB TMP "${gloo_SOURCE_DIR}/${pattern}")
    list(APPEND GLOO_HEADERS ${TMP})
endforeach()

list(REMOVE_ITEM GLOO_HEADERS "${gloo_SOURCE_DIR}/gloo/rendezvous/redis_store.h")

# Copy headers preserving directory structure
foreach(hdr ${GLOO_HEADERS})
    file(RELATIVE_PATH rel_path ${gloo_SOURCE_DIR} ${hdr})
    get_filename_component(dir ${rel_path} DIRECTORY)
    file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/gloo-headers/${dir})
    file(COPY ${hdr} DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/gloo-headers/${dir})
endforeach()

set(GLOO_INCLUDE_DIR ${CMAKE_CURRENT_BINARY_DIR}/gloo-headers)

message(STATUS "GLOO_INCLUDE_DIR: ${GLOO_INCLUDE_DIR}")

# Test it
# add_executable(main main.cpp)
# target_include_directories(main PUBLIC ${GLOO_INCLUDE_DIR})