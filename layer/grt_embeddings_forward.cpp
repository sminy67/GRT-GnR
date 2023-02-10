/*
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*/

#include <ATen/ATen.h>
#include <torch/extension.h>

using namespace at;

Tensor grt_embeddings_forward_cuda(
    int32_t num_group_bags,
    int32_t num_bags,
    int32_t emb_dims,
    const std::vector<int>& row_shapes,
    const std::vector<int>& emb_shapes,
    const std::vector<int>& tt_ranks,
    Tensor tt_idx_shapes,
    int32_t intra_nnz,
    Tensor intra_group_indices,
    Tensor intra_group_offsets,
    Tensor inter_group_indices,
    Tensor inter_group_offsets,
    const std::vector<Tensor>& tt_cores);

Tensor tt_embeddings_forward_cuda(
    int32_t num_bags,
    int32_t emb_dims,
    const std::vector<int>& row_shapes,
    const std::vector<int>& emb_shapes,
    const std::vector<int>& tt_ranks,
    Tensor tt_idx_shapes,
    int32_t nnz,
    Tensor indices,
    Tensor offsets,
    const std::vector<Tensor>& tt_cores);

PYBIND11_MODULE(grt_embeddings_forward, m) {
  m.def("grt_forward", &grt_embeddings_forward_cuda, "grt_forward()");
  m.def("tt_forward", &tt_embeddings_forward_cuda, "tt_forward()");
}
