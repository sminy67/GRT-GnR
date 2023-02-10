#!/usr/bin/env /home/sminyu/venv/rec_sys/bin/python3
import numpy as np
import torch
import torch.nn as nn
from layer.grt_embeddings import GRTEmbeddingBag
import time
import argparse
import os
from datasets.data_utils import load_data
from scipy.sparse import coo_matrix, csr_matrix

def main(args):
    dataloader, num_items = load_data(args)
    
    grt_emb = GRTEmbeddingBag(args,
                              num_rows=num_items,
                              emb_dims=64,
                              tt_ranks=[32, 32],
                              row_shapes=[120, 140, 195],
                              emb_shapes=[4, 4, 4]).cuda()

    if args.grouping:
        with torch.no_grad():
            for batch_idx, (intra_group_offsets, intra_group_indices, inter_group_offsets, inter_group_indices) in enumerate(dataloader):
                indices = (intra_group_indices.cuda(), inter_group_indices.cuda())
                offsets = (intra_group_offsets.cuda(), inter_group_offsets.cuda())
                output = grt_emb(indices, offsets)
                if batch_idx == 3:
                    exit()
    else:
        with torch.no_grad():
            for batch_idx, (offsets, indices) in enumerate(dataloader):
                indices = (indices.cuda(), None)
                offsets = (offsets.cuda(), None)
                output = grt_emb(indices, offsets)
                if batch_idx == 3:
                    exit()

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Profiling TT Embedding Table")

    parser.add_argument("--data-path", type=str, default="/home/sminyu/rec_sys/TT_RnG/data/raw_data/Movies_and_TV.csv")
    parser.add_argument("--thres", type=float, default=4.0)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--grouping", type=int, default=0)
    parser.add_argument("--group-size", type=int, default=50)
    parser.add_argument("--use-group-alg", type=int, default=0)
    parser.add_argument("--alg-name", type=str, default="random", choices=["random", "sort", "optim"])

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = '7'
    main(args)
