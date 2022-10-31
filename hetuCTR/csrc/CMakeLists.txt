aux_source_directory(utils utils_src)
aux_source_directory(pybind pybind_src)
aux_source_directory(core core_src)

pybind11_add_module(hetuCTR ${pybind_src} ${utils_src} ${core_src})

target_include_directories(hetuCTR
    PRIVATE
    .
    ${cub_SOURCE_DIR}
    ${thrust_SOURCE_DIR}
    ${NCCL_INCLUDE_DIRS}
)

target_link_libraries(hetuCTR
    PRIVATE
    CUDA::cudart
    ${NCCL_LIBRARIES}
    ZMQ::lib
)

find_package(OpenMP REQUIRED)
aux_source_directory(partition partition_src)
pybind11_add_module(hetuCTR_partition ${partition_src})
target_link_libraries(hetuCTR_partition PRIVATE OpenMP::OpenMP_CXX)
target_include_directories(hetuCTR_partition PRIVATE . partition)
