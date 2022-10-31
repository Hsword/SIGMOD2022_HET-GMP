#pragma once

#include "core/types.h"
#include "pybind/pybind.h"

namespace hetuCTR {

class PartitionStruct {
public:
  PartitionStruct(const py::array_t<int>& _input_data, const py::array_t<float>& _comm_mat, int _n_part, int _batch_size);
  void refineData();
  void refineEmbed();
  py::array_t<float> getCommunication();
  py::array_t<float> getPriority();

  std::vector<int> res_data_, res_embed_;
  std::vector<int> cnt_data_, cnt_embed_;
private:
  float alpha_, beta_;

  int n_part_, n_data_, n_slot_, n_edge_, n_embed_;
  int batch_size_;
  std::vector<int> embed_indptr_, embed_indices_, data_indptr_, data_indices_;
  std::vector<float> soft_cnt_, embed_weight_;
  std::vector<std::vector<int>> cnt_part_embed_;

  std::vector<std::vector<float>> comm_mat_;
};

std::unique_ptr<PartitionStruct> partition(
  const py::array_t<int>& _input_data,
  const py::array_t<float>& _comm_mat,
  int n_part, int batch_size
);


} // namespace hetuCTR
