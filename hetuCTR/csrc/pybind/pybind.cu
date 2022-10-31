#include "pybind.h"

#include "container.h"
#include "utils/initializer.h"
#include "utils/thread_pool.h"

using namespace hetuCTR;

static std::unique_ptr<TableContainer> makeTable(
  const int rank,
  const int nrank,
  const int device_id,
  const std::string &ip,
  const int port,
  const size_t embedding_length,
  const size_t embedding_width,
  const version_t pull_bound,
  const version_t push_bound,
  py::array_t<worker_t> root_id_arr,
  py::array_t<index_t> storage_id_arr,
  const Initializer &init,
  const embed_t learning_rate,
  const int verbose)
{
  PYTHON_CHECK_ARRAY(root_id_arr);
  PYTHON_CHECK_ARRAY(storage_id_arr);
  SArray<worker_t> root_id_arr_shared(root_id_arr.mutable_data(), root_id_arr.size());
  SArray<index_t> storage_id_arr_shared(storage_id_arr.mutable_data(), storage_id_arr.size());
  return std::make_unique<TableContainer>(
    rank, nrank, device_id,
    ip, port,
    embedding_length, embedding_width,
    pull_bound, push_bound,
    root_id_arr_shared, storage_id_arr_shared,
    init, learning_rate, verbose);
}

static std::unique_ptr<Initializer> makeInitializer(InitType type, float param_a, float param_b) {
  return std::make_unique<Initializer>(type, param_a, param_b);
}

PYBIND11_MODULE(hetuCTR, m) {
  m.doc() = "hetuCTR C++/CUDA Implementation"; // module docstring

  // Used for async call
  typedef std::future<void> wait_t;
  py::class_<wait_t>(m, "_waittype")
  .def("wait", [](const wait_t &w){
    py::gil_scoped_release release;
    w.wait();
  });

  py::enum_<InitType>(m, "InitType", py::module_local())
    .value("Zero", InitType::kZero)
    .value("Normal", InitType::kNormal)
    .value("TruncatedNormal", InitType::kTruncatedNormal)
    .value("Uniform", InitType::kUniform);

  py::class_<Initializer, std::unique_ptr<Initializer>>(m, "Initializer", py::module_local())
    .def(py::init(&makeInitializer));

  py::class_<TableContainer, std::unique_ptr<TableContainer>>(m, "HetuTable", py::module_local())
    .def(py::init(&makeTable),
      py::arg("rank"), py::arg("nrank"), py::arg("device_id"),
      py::arg("ip"), py::arg("port"),
      py::arg("length"), py::arg("width"),
      py::arg("pull_bound"), py::arg("push_bound"),
      py::arg("root_arr"), py::arg("storage_arr"),
      py::arg("init"), py::arg("learning_rate"), py::arg("verbose"))
    .def("preprocess", &TableContainer::preprocess)
    .def("push_pull", &TableContainer::pushPull)
    .def("async_push_pull", &TableContainer::pushPullAsync)
    .def("async_preprocess", &TableContainer::preprocessAsync)
    .def("__repr__", &TableContainer::debugString)
    .def("debug",  &TableContainer::debugStringFull);

  m.def("seed", [](unsigned long seed){ __seed = seed; });

} // PYBIND11_MODULE
