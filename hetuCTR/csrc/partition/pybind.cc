#include "pybind/pybind.h"
#include "partition.h"

namespace hetuCTR {

PYBIND11_MODULE(hetuCTR_partition, m) {
  m.doc() = "hetuCTR graph partition C++ implementation"; // module docstring
  py::class_<PartitionStruct>(m, "_PartitionStruct", py::module_local())
    .def("refine_data", &PartitionStruct::refineData)
    .def("refine_embed", &PartitionStruct::refineEmbed)
    .def("get_communication", &PartitionStruct::getCommunication)
    .def("get_priority", &PartitionStruct::getPriority)
    .def("get_result", [](PartitionStruct &s) {
      return py::make_tuple(bind::vec_nocp(s.res_data_), bind::vec_nocp(s.res_embed_));
    })
    .def("get_data_cnt", [](PartitionStruct &s) {
      return bind::vec_nocp(s.cnt_data_);
    })
    .def("get_embed_cnt", [](PartitionStruct &s) {
      return bind::vec_nocp(s.cnt_embed_);
    });
  m.def("partition", partition);

} // PYBIND11_MODULE

} // namespace hetuCTR
