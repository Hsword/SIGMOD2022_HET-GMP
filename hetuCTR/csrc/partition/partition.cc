#include "partition.h"

#include <bits/stdc++.h>

namespace hetuCTR {

static float quickpow(float a,int n) {
    float ans = 1, temp = a;
    while(n) {
      if(n&1) ans *= temp;
      n >>= 1;
      temp *= temp;
    }
    return ans;
}

template<typename T>
int argmax(const std::vector<T> &val) {
  int arg_res = 0;
  for (int i = 1 ; i < (int)val.size(); i++)
    if (val[i] > val[arg_res])
      arg_res = i;
  return arg_res;
}

PartitionStruct::PartitionStruct(const py::array_t<int>& _input_data, const py::array_t<float>& _comm_mat, int _n_part, int _batch_size)
: n_part_(_n_part), batch_size_(_batch_size) {
  n_data_ = _input_data.shape(0);
  n_slot_ = _input_data.shape(1);
  n_edge_ = n_data_ * n_slot_;
  const int *data = _input_data.data();
  n_embed_ = 0;
  for (int i = 0 ; i < n_edge_; i++) {
    n_embed_ = std::max(n_embed_, data[i]);
  }
  n_embed_++;
  std::vector<int> count(n_embed_);
  data_indptr_.resize(n_data_ + 1);
  embed_indptr_.resize(n_embed_ + 1);
  data_indices_.resize(n_edge_);
  embed_indices_.resize(n_edge_);

  for (int i = 0; i <= n_data_; i++) data_indptr_[i] = i * n_slot_;
  for (int i = 0 ; i < n_edge_; i++) {
    count[data[i]]++;
    data_indices_[i] = data[i];
  }
  for (int i = 1;i <= n_embed_; i++) {
    embed_indptr_[i] = embed_indptr_[i-1] + count[i - 1];
    count[i - 1] = 0;
  }
  assert(embed_indptr_[n_embed_] == n_edge_);
  for (int i = 0 ; i < n_edge_; i++) {
    int data_id = i / n_slot_;
    int embed_id = data[i];
    embed_indices_[embed_indptr_[embed_id] + count[embed_id]] = data_id;
    count[embed_id]++;
  }
  //  initSoftLabel
  int max = n_data_ / n_part_;
  soft_cnt_.resize(n_data_, 1);
  for (int i = 0; i < max; i++) {
    soft_cnt_[i] = 1 - quickpow(1.0 - (float)i / (float)max, batch_size_);
    soft_cnt_[i] *= (float)max / (float)batch_size_;
  }
  // initResult
  res_data_.resize(n_data_);
  res_embed_.resize(n_embed_);
  cnt_data_.resize(n_part_, 0);
  cnt_embed_.resize(n_part_, 0);

  cnt_part_embed_.resize(n_part_, std::vector<int>(n_embed_, 0));

  for (int i = 0; i < n_data_; i++) {
    res_data_[i] = rand() % n_part_;
    cnt_data_[res_data_[i]]++;
    for (int j = data_indptr_[i]; j < data_indptr_[i+1]; j++) {
      cnt_part_embed_[res_data_[i]][data_indices_[j]]++;
    }
  }
  for (int i = 0; i < n_embed_; i++) {
    res_embed_[i] = rand() % n_part_;
    cnt_embed_[res_embed_[i]]++;
  }
  // init param
  alpha_ = -100.0 / (n_data_ / n_part_);
  beta_ = -100.0 / (n_embed_ / n_part_);
  // init communication matrix
  comm_mat_.resize(n_part_, std::vector<float>(n_part_));
  for (int i = 0; i < n_part_; i++)
    for (int j = 0; j < n_part_; j++) comm_mat_[i][j] = _comm_mat.at(i, j);
}

void PartitionStruct::refineData() {
  #pragma omp parallel for num_threads(16)
  for (int i = 0; i < n_data_; i++) {
    std::vector<float> score(n_part_, 0);
    int old = res_data_[i];
    for (int j = 0; j < n_part_; j++) {
      int cnt_data = old == j ? cnt_data_[j] - 1 : cnt_data_[j];
      score[j] = alpha_ * cnt_data;
    }
    for (int j = data_indptr_[i]; j < data_indptr_[i+1]; j++) {
      int embed_id = data_indices_[j];
      int belong = res_embed_[embed_id];
      for (int k = 0; k < n_part_; k++) {
        int use_cnt = k == old ? cnt_part_embed_[k][embed_id] - 1 : cnt_part_embed_[k][embed_id];
        score[k] -= (soft_cnt_[use_cnt + 1] - soft_cnt_[use_cnt]) * comm_mat_[k][belong];
      }
    }
    int s = argmax(score);
    #pragma omp critical
    if (s != old) {
      cnt_data_[old]--;
      cnt_data_[s]++;
      for (int j = data_indptr_[i]; j < data_indptr_[i+1]; j++) {
        cnt_part_embed_[s][data_indices_[j]]++;
        cnt_part_embed_[old][data_indices_[j]]--;
      }
      res_data_[i] = s;
    }
  }
}

void PartitionStruct::refineEmbed() {
  embed_weight_.clear();
  embed_weight_.resize(n_part_, 0);
  for (int i = 0; i < n_part_; i++) {
    for (int j = 0; j < n_embed_; j++) {
      if (res_embed_[j] != i) embed_weight_[res_embed_[j]] += soft_cnt_[cnt_part_embed_[i][j]];
    }
  }
  for (int i = 0; i < n_part_; i++)
    embed_weight_[i] *= float(batch_size_ * n_part_) / n_data_;

  #pragma omp parallel for num_threads(16)
  for (int i = 0; i < n_embed_; i++) {
    std::vector<float> score(n_part_), cnt(n_part_, 0);
    int old = res_embed_[i];
    for (int j = embed_indptr_[i]; j < embed_indptr_[i+1]; j++) {
      cnt[res_data_[embed_indices_[j]]]++;
    }
    for (int j = 0; j < n_part_; j++) {
      int cnt_embed = old == j ? cnt_embed_[j] - 1 : cnt_embed_[j];
      score[j] = beta_ * cnt_embed;
    }
    for (int j = 0; j < n_part_; j++) {
      for (int k = 0; k < n_part_; k++)
        score[k] -= comm_mat_[j][k] * soft_cnt_[cnt[j]];
      score[j] -= 0.01 * embed_weight_[j] * soft_cnt_[embed_indptr_[i+1]-embed_indptr_[i]];
    }
    int s = argmax(score);
    #pragma omp critical
    if (s != old) {
      cnt_embed_[old]--;
      cnt_embed_[s]++;
      res_embed_[i] = s;
      for (int j = 0; j < n_part_; j++) {
        if (j != s)
          embed_weight_[s] += soft_cnt_[cnt_part_embed_[j][i]] * (batch_size_ * n_part_) / n_data_;
        if (j != old)
          embed_weight_[old] -= soft_cnt_[cnt_part_embed_[j][i]] * (batch_size_ * n_part_) / n_data_;
      }
    }
  }
}

py::array_t<float> PartitionStruct::getCommunication() {
  py::array_t<float> cost({n_part_, n_part_});
  for (int i = 0; i < n_part_; i++)
    for (int j = 0; j < n_part_; j++)
      cost.mutable_at(i, j) = 0;
  for (int i = 0; i < n_part_; i++)
    for (int j = 0; j < n_embed_; j++)
      cost.mutable_at(i, res_embed_[j]) += soft_cnt_[cnt_part_embed_[i][j]];
  for (int i = 0; i < n_part_; i++)
    for (int j = 0; j < n_part_; j++)
      cost.mutable_at(i, j) *= float(batch_size_ * n_part_) / n_data_;
  return cost;
}

py::array_t<float> PartitionStruct::getPriority() {
  py::array_t<float> priority({n_part_, n_embed_});
  for (int i = 0; i < n_part_; i++) {
    for (int j = 0 ; j < n_embed_; j++) {
      if (cnt_part_embed_[i][j] == 0) priority.mutable_at(i, j) = 0;
      else
        priority.mutable_at(i, j) = comm_mat_[i][res_embed_[j]] * std::pow(soft_cnt_[cnt_part_embed_[i][j]], 2l) *
          ((1.0 / (embed_indptr_[j + 1] - embed_indptr_[j])) + (1.0 / cnt_part_embed_[i][j]));
    }
  }
  return priority;
}

std::unique_ptr<PartitionStruct> partition(
  const py::array_t<int>& _input_data,
  const py::array_t<float>& _comm_mat,
  int n_part, int batch_size) {
  PYTHON_CHECK_ARRAY(_input_data);
  PYTHON_CHECK_ARRAY(_comm_mat);
  assert(_input_data.ndim() == 2);
  assert(_comm_mat.ndim() == 2);
  assert(_comm_mat.shape(0) == _comm_mat.shape(1));
  assert(_comm_mat.shape(0) == n_part);
  return  std::make_unique<PartitionStruct>(_input_data, _comm_mat, n_part, batch_size);
}

}
