# Try to find NCCL
#
# The following variables are optionally searched for defaults
#  NCCL_ROOT: Base directory where all NCCL components are found
#  NCCL_ROOT_DIR: Base directory where all NCCL components are found
#  NCCL_INCLUDE_DIR: Directory where NCCL header is found
#  NCCL_LIB_DIR: Directory where NCCL library is found
#
# The following are set after configuration is done:
#  NCCL_FOUND
#  NCCL_INCLUDE_DIRS
#  NCCL_LIBRARIES
#
# The path hints include CUDA_TOOLKIT_ROOT_DIR seeing as some folks
# install NCCL in the same location as the CUDA toolkit.
# See https://github.com/caffe2/caffe2/issues/1601

set(NCCL_ROOT_DIR $ENV{NCCL_ROOT_DIR} CACHE PATH "Folder contains NVIDIA NCCL")

find_path(NCCL_INCLUDE_DIRS
    NAMES nccl.h
    HINTS
    ${NCCL_ROOT}
    ${NCCL_ROOT}/include
    ${NCCL_INCLUDE_DIR}
    ${NCCL_ROOT_DIR}
    ${NCCL_ROOT_DIR}/include
    ${CUDA_TOOLKIT_ROOT_DIR}/include)

if ($ENV{USE_STATIC_NCCL})
    message(STATUS "USE_STATIC_NCCL detected. Linking against static NCCL library")
    set(NCCL_LIBNAME "libnccl_static.a")
else()
    set(NCCL_LIBNAME "nccl")
endif()

find_library(NCCL_LIBRARIES
    NAMES ${NCCL_LIBNAME}
    HINTS
    ${NCCL_LIB_DIR}
    ${NCCL_ROOT}
    ${NCCL_ROOT}/lib
    ${NCCL_ROOT}/lib/x86_64-linux-gnu
    ${NCCL_ROOT}/lib64
    ${NCCL_ROOT_DIR}
    ${NCCL_ROOT_DIR}/lib
    ${NCCL_ROOT_DIR}/lib/x86_64-linux-gnu
    ${NCCL_ROOT_DIR}/lib64
    ${CUDA_TOOLKIT_ROOT_DIR}/lib64)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(NCCL DEFAULT_MSG NCCL_INCLUDE_DIRS NCCL_LIBRARIES)

if (NCCL_FOUND)
    set(NCCL_HEADER_FILE "${NCCL_INCLUDE_DIRS}/nccl.h")
    message(STATUS "Determining NCCL version from the header file: ${NCCL_HEADER_FILE}")
    file (STRINGS ${NCCL_HEADER_FILE} NCCL_VERSION_DEFINED
        REGEX "^[ \t]*#define[ \t]+NCCL_VERSION_CODE[ \t]+[0-9]+.*$" LIMIT_COUNT 1)
    if (NCCL_VERSION_DEFINED)
        string (REGEX REPLACE "^[ \t]*#define[ \t]+NCCL_VERSION_CODE[ \t]+" ""
            NCCL_VERSION ${NCCL_VERSION_DEFINED})
        message(STATUS "NCCL_VERSION_CODE: ${NCCL_VERSION}")
    endif()
    if ((NOT NCCL_VERSION_DEFINED) OR NCCL_VERSION LESS 2708)
        message(FATAL_ERROR "Required NCCL version >= 2708(2.7.8), while current NCCL version is ${NCCL_VERSION}.")
    endif()
    set(NCCL_LIBRARIES ${NCCL_LIBRARIES})
    message(STATUS "Found NCCL (include: ${NCCL_INCLUDE_DIRS}, library: ${NCCL_LIBRARIES})")
    mark_as_advanced(NCCL_ROOT_DIRS NCCL_INCLUDE_DIRS NCCL_LIBRARIES)
endif()