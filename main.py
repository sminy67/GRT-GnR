#!/usr/bin/env /home/sminyu/venv/rec_sys/bin/python3
import numpy as np
import torch
import torch.nn as nn
from layer.grt_embeddings import GRTEmbeddingBag
import time
import random
import argparse
import logging
import os
from dataset import load_data, count_cached_item_access, count_item_access
from scipy.sparse import coo_matrix, csr_matrix

logger = logging.getLogger(__name__)

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    if torch.cuda.device_count() > 0:
        torch.cuda.manual_seed_all(seed)

def main(args):
    set_seed(0)
    if args.analyze_data:
        indices, offsets, num_items, num_users = load_data(args)
        #count_cached_item_access(indices, num_items, args)
        count_item_access(indices, num_items, args)

        exit()
    else:    
        dataloader, num_items, row_shapes, emb_shapes = load_data(args)

    '''
    grt_emb = GRTEmbeddingBag(args,
                              num_rows=num_items,
                              emb_dims=args.emb_dims,
                              tt_ranks=[32, 32],
                              row_shapes=row_shapes,
                              emb_shapes=emb_shapes).cuda()
    '''

    emb = nn.EmbeddingBag(num_embeddings=num_items,
                          embedding_dim=args.emb_dims,
                          mode="sum").cuda()

    if args.grouping:
        with torch.no_grad():
            total_time = 0
            num_tt_gather = 0
            total_intra_group_idx = 0
            for batch_idx, (intra_group_offsets, intra_group_indices, inter_group_offsets, inter_group_indices) in enumerate(dataloader):
                indices = (intra_group_indices.cuda(), inter_group_indices.cuda())
                offsets = (intra_group_offsets.cuda(), inter_group_offsets.cuda())
                num_tt_gather += inter_group_indices.shape[0]
                total_intra_group_idx += intra_group_indices.shape[0]

                start = time.time()
                output = grt_emb(indices, offsets)
                torch.cuda.synchronize()
                end = time.time()
                if (batch_idx > 9):
                    total_time += (end - start)
            print(f"Total time : {total_time/(len(dataloader)-10)}")
            print(f"Total TT-gather : {num_tt_gather}")
            print(f"Total Intra Group Indices: {total_intra_group_idx}")
    else:
        with torch.no_grad():
            total_time = 0
            num_tt_gather = 0
            for batch_idx, (offsets, indices) in enumerate(dataloader):
                num_tt_gather += indices.shape[0]
                #indices = (indices.cuda(), None)
                #offsets = (offsets.cuda(), None)
                indices = indices.cuda()
                offsets = offsets.cuda()

                start = time.time()
                #output = grt_emb(indices, offsets)
                output = emb(indices, offsets)
                torch.cuda.synchronize()
                end = time.time()
                if (batch_idx > 9):
                    total_time += (end - start)
            print(f"Total time : {total_time/(len(dataloader)-10)}")
            print(f"Total TT-gather : {num_tt_gather}")

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Profiling TT Embedding Table")

    parser.add_argument("--path", type=str, default="/home/sminyu/rec_sys/data/")
    parser.add_argument("--data-name", type=str, default="Movies_and_TV")
    parser.add_argument("--thres", type=float, default=4.0)
    parser.add_argument("--analyze-data", type=int, default=0)
    parser.add_argument("--use-cache", type=int, default=0)
    parser.add_argument("--cache-size", type=float, default=0.01)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--grouping", type=int, default=0)
    parser.add_argument("--num-cores", type=int, default=3)
    parser.add_argument("--emb-dims", type=int, default=64)
    parser.add_argument("--use-group-alg", type=int, default=0)
    parser.add_argument("--alg-name", type=str, default="optim", choices=["random", "sort", "optim"])

    args = parser.parse_args()
    args.data_path = args.path + args.data_name + ".csv"
    os.environ["CUDA_VISIBLE_DEVICES"] = '4'

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)
    main(args)
