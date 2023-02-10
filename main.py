#!/usr/bin/env /home/sminyu/venv/rec_sys/bin/python3
import numpy as np
import torch
import torch.nn as nn
from layer.grt_embeddings import GRTEmbeddingBag
import time
import random
import argparse
import os
from dataset import load_data
from scipy.sparse import coo_matrix, csr_matrix

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    if torch.cuda.device_count() > 0:
        torch.cuda.manual_seed_all(seed)

def main(args):
    set_seed(0)
    dataloader, num_items, row_shapes, emb_shapes = load_data(args)
    
    grt_emb = GRTEmbeddingBag(args,
                              num_rows=num_items,
                              emb_dims=args.emb_dims,
                              tt_ranks=[32, 32],
                              row_shapes=row_shapes,
                              emb_shapes=emb_shapes).cuda()

    if args.grouping:
        with torch.no_grad():
            total_time = 0
            for batch_idx, (intra_group_offsets, intra_group_indices, inter_group_offsets, inter_group_indices) in enumerate(dataloader):
                indices = (intra_group_indices.cuda(), inter_group_indices.cuda())
                offsets = (intra_group_offsets.cuda(), inter_group_offsets.cuda())
                start = time.time()
                output = grt_emb(indices, offsets)
                torch.cuda.synchronize()
                end = time.time() - start
                if batch_idx > 8:
                    total_time += end
            print(f"Total time : {total_time/(len(dataloader)-10)}")
    else:
        with torch.no_grad():
            total_time = 0
            for batch_idx, (offsets, indices) in enumerate(dataloader):
                indices = (indices.cuda(), None)
                offsets = (offsets.cuda(), None)
                start = time.time()
                output = grt_emb(indices, offsets)
                torch.cuda.synchronize()
                end = time.time() - start
                if batch_idx > 8:
                    total_time += end
            print(f"Total time : {total_time/(len(dataloader)-10)}")
            

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Profiling TT Embedding Table")

    parser.add_argument("--data-path", type=str, default="/home/sminyu/rec_sys/data/Movies_and_TV.csv")
    parser.add_argument("--thres", type=float, default=4.0)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--grouping", type=int, default=0)
    parser.add_argument("--num-cores", type=int, default=3)
    parser.add_argument("--emb-dims", type=int, default=64)
    parser.add_argument("--use-group-alg", type=int, default=0)
    parser.add_argument("--alg-name", type=str, default="sort", choices=["random", "sort", "optim"])

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = '7'
    main(args)
