#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import grt_embeddings_forward
import torch.nn as nn
import time

class GRTEmbeddingBag(nn.Module):
    __constants__ = ["num_rows, emb_dims, tt_shapes, tt_ranks"]
    
    def __init__(
        self,
        args,
        num_rows: int,
        num_cache: int,
        emb_dims: int,
        tt_ranks: List[int],
        row_shapes: Optional[List[int]] = None,
        emb_shapes: Optional[List[int]] = None,
        weight_dist: str = "approx-normal",
    ) -> None:
        super(GRTEmbeddingBag, self).__init__()
        
        self.num_rows = num_rows
        self.emb_dims = emb_dims
        self.tt_ranks = [1] + tt_ranks + [1]
        self.row_shapes = row_shapes
        self.emb_shapes = emb_shapes
        
        self.check_config_is_available()
        
        self.num_cores = len(tt_ranks) + 1
        self.grouping = False
        if args.grouping:
            self.group_size = self.row_shapes[-1]
            self.num_groups = num_rows // self.row_shapes[-1]
            self.grouping = True

        self.create_tt_idx_shape()
        self.create_tt_params(weight_dist)

        self.cache_emb = nn.EmbeddingBag(num_embeddings=num_cache,
                                         embedding_dim=emb_dims,
                                         include_last_offset=True)

    def check_config_is_available(self):
        
        assert torch.cuda.is_available()
        assert self.num_rows > 0
        assert self.emb_dims > 0
        assert len(self.row_shapes) >= 2
        assert len(self.row_shapes) <= 3
        assert len(self.emb_shapes) == len(self.row_shapes)
        assert len(self.tt_ranks) - 1 == len(self.row_shapes)
        
        assert all(fac > 0 for fac in self.row_shapes)
        assert all(fac > 0 for fac in self.emb_shapes)
        assert all(rank > 0 for rank in self.tt_ranks)
                    
    def create_tt_idx_shape(self):

        tt_idx_shape = []
        tt_idx_value = 1
        
        if self.grouping:
            for n in range(self.num_cores - 1):
                tt_idx_shape.append(tt_idx_value)
                tt_idx_value *= self.row_shapes[self.num_cores - n - 2]
        else:
            for n in range(self.num_cores):
                tt_idx_shape.append(tt_idx_value)
                tt_idx_value *= self.row_shapes[self.num_cores - n - 1]

        tt_idx_shape.reverse()
        self.register_buffer("tt_idx_shapes", torch.tensor(tt_idx_shape, dtype=torch.long))        

    def create_tt_params(self, weight_dist):
        
        self.tt_cores = torch.nn.ParameterList()
        for i in range(self.num_cores):
            self.tt_cores.append(torch.nn.Parameter(torch.empty(
                [self.row_shapes[i], self.tt_ranks[i] * self.emb_shapes[i] * self.tt_ranks[i + 1]],
                device=torch.cuda.current_device(),
                dtype=torch.float32)))
        self.reset_parameters(weight_dist)

    def reset_parameters(self, weight_dist: str) -> None:
        
        assert weight_dist in [
            "uniform",
            "naive-uniform",
            "normal",
            "approx-normal",
            "constant"
        ]
        
        if weight_dist == "uniform":
            lamb = 2.0 / (self.num_rows + self.emb_dims)
            stddev = np.sqrt(lamb)
            tt_ranks = np.array(self.tt_ranks)
            cr_exponent = -1.0 / (2 * self.num_cores)
            var = np.prod(tt_ranks**cr_exponent)
            core_stddev = stddev ** (1.0 / self.num_cores) * var
            for i in range(self.num_cores):
                nn.init.uniform_(self.tt_cores[i], 0.0, core_stddev)
        elif weight_dist == "naive-uniform":
            for i in range(self.num_cores):
                nn.init.uniform_(self.tt_cores[i], 0.0, 1 / np.sqrt(self.num_rows))
        elif weight_dist == "normal":
            mu = 0.0
            sigma = 1.0 / np.sqrt(self.num_rows)
            scale = 1.0 / self.tt_ranks[0]
            for i in range(self.num_cores):
                nn.init.normal_(self.tt_cores[i], mu, sigma)
                self.tt_cores[i].data *= scale
        elif weight_dist == "approx-normal":
            mu = 0.0
            sigma = 1.9
            scale = np.power(1 / np.sqrt(3 * self.num_rows), 1/3)
            for i in range(self.num_cores):
                W = np.random.normal(loc=mu, scale=sigma, size=np.asarray(self.tt_cores[i].shape)).astype(np.float32)
                core_shape = self.tt_cores[i].shape
                W = W.flatten()
                for ele in range(W.shape[0]):
                    while np.abs(W[ele]) < 2:
                        W[ele] = np.random.normal(loc=mu, scale=sigma, size=[1]).astype(np.float32)
                W = np.reshape(W, core_shape)
                W *= scale
                self.tt_cores[i].data = torch.tensor(W, requires_grad=True)
        elif weight_dist == "constant":
            for i in range(self.num_cores):
                core_shape = self.tt_cores[i].shape
                W = torch.arange(core_shape[0])
                W = W.unsqueeze(0).T
                W = W.repeat(1, core_shape[1]).type(dtype=torch.float32)
                self.tt_cores[i].data = W.clone().detach().requires_grad_(True)
                
    def forward(self, indices: Tuple, offsets: Tuple, cached_indices: torch.Tensor, cached_offsets: torch.Tensor) -> torch.Tensor:
        if self.grouping:
            intra_group_indices, inter_group_indices = indices
            intra_group_offsets, inter_group_offsets = offsets
            num_group_bags = inter_group_indices.numel()
            num_bags = inter_group_offsets.numel() - 1
            num_indices = intra_group_indices.numel()

            output = self.cache_emb(cached_indices, cached_offsets)
            
            output = grt_embeddings_forward.grt_forward(
                num_group_bags,
                num_bags,
                self.emb_dims,
                self.row_shapes,
                self.emb_shapes,
                self.tt_ranks,
                self.tt_idx_shapes,
                num_indices,
                intra_group_indices,
                intra_group_offsets,
                inter_group_indices,
                inter_group_offsets,
                self.tt_cores,
                output)
        else:
            indices, _ = indices
            offsets, _ = offsets
            num_bags = offsets.numel() - 1
            num_indices = indices.numel()
            
            start = time.time() 
            output = self.cache_emb(cached_indices, cached_offsets)
            end = time.time() - start
            torch.cuda.synchronize()
            print(f"Cache Latency: {end}")
            
            start = time.time()
            output = grt_embeddings_forward.tt_forward(
                num_bags,
                self.emb_dims,
                self.row_shapes,
                self.emb_shapes,
                self.tt_ranks,
                self.tt_idx_shapes,
                num_indices,
                indices,
                offsets,
                self.tt_cores,
                output)
            torch.cuda.synchronize()
            end = time.time() - start
            print(f"GRT-GnR Latency: {end}")
        return output
        
