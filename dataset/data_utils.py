import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.sparse import coo_matrix, csr_matrix
import scipy.sparse as sp
from utils.utils import factorize_shapes
from .grouping.optimize import optim_indices
import seaborn as sns
import logging
import os

logger = logging.getLogger(__name__)

def load_data(args):
    data_path = f"/home/sminyu/rec_sys/saved_data/{args.data_name}/sparse_data.npz"
    if os.path.exists(data_path):
        sparse_data = sp.load_npz(data_path)
    else:
        logger.info(f"Loading {args.data_name} Dataset")
        
        col = ['user_id', 'item_id', 'rating', 'timestamp']
        data = pd.read_csv(args.data_path, names=col)
        
        # Implicit Threshold
        data.rating = [1 if x>=args.thres else 0 for x in data.rating.tolist()]
        data = data[data.rating==1]

        user_list = list(data['user_id'].unique())
        item_list = list(data['item_id'].unique())
        
        num_users = len(user_list)
        num_items = len(item_list)
        
        user_id_dict = {old_uid: new_uid for new_uid, old_uid in enumerate(user_list)}
        data.user_id = [user_id_dict[x] for x in data.user_id.tolist()]
        
        item_id_dict = {old_uid: new_uid for new_uid, old_uid in enumerate(item_list)}
        data.item_id = [item_id_dict[x] for x in data.item_id.tolist()]

        data = data[['user_id', 'item_id', 'rating']]
        data = data.drop_duplicates()
        data = data.sort_values(by='user_id', ascending=True)
        
        row_shapes = factorize_shapes(n=num_items, d=3)
        emb_shapes = factorize_shapes(n=args.emb_dims, d=3)
        args.group_size = row_shapes[-1]

        row = data.user_id.values
        col = data.item_id.values
        val = data.rating.values
        
        sparse_data = coo_matrix((val, (row, col)), shape=(num_users, num_items)).tocsr()
        sp.save_npz(f"/home/sminyu/rec_sys/saved_data/{args.data_name}/sparse_data.npz", sparse_data)
    
    if args.use_cache:
        cache_size = int(args.cache_size * num_items * 0.01)
        saved_data_path = f"/home/sminyu/rec_sys/saved_data/use_cache_{args.cache_size}_sparse_data.npz"
        saved_cache_path = f"/home/sminyu/rec_sys/saved_data/use_cache_{args.cache_size}_sparse_cache.npz"
        if os.path.exists(saved_cache_path) and os.path.exists(saved_data_path):
            sparse_data = sp.load_npz(saved_data_path)
            sparse_cache = sp.load_npz(saved_cache_path)
            sparse_cache = sparse_cache.tocsr()
        else:
            sparse_cache, sparse_data = gen_cache(sparse_data, cache_size, args)

        cache_indices = sparse_cache.indices
        cache_offsets = sparse_cache.indptr
        
    indices = sparse_data.indices
    offsets = sparse_data.indptr

    if args.analyze_data:
        return indices, offsets, num_items, num_users
   
    if args.use_group_alg:
        if args.alg_name == "random":
            indices = random_indices(indices, num_items)
        elif args.alg_name == "sort":
            indices = sort_indices(indices, num_items)
        elif args.alg_name == "optim":
            optim_name = f"/home/sminyu/rec_sys/GRT-GnR/dataset/grouping/optim_data/{args.data_name}.npy"
            if os.path.exists(optim_name):
                idx = np.load(optim_name)
                indices = optimize_indices(indices, idx)
            else:
                indices = optim_indices(indices, sparse_data, args.group_size, args)
    
    if args.use_cache:
        dataset = RecDataset(torch.tensor(offsets, dtype=torch.long), torch.tensor(indices, dtype=torch.long),
                             torch.tensor(cache_offsets, dtype=torch.long), torch.tensor(cache_indices, dtype=torch.long), args)
        if args.grouping:
            dataloader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=cached_grouped_collate_fn, shuffle=False)
        else:
            dataloader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=cached_collate_fn, shuffle=False)
        
        return dataloader, num_items, row_shapes, emb_shapes, cache_size
    else:
        dataset = RecDataset(torch.tensor(offsets, dtype=torch.long), torch.tensor(indices, dtype=torch.long),
                             None, None, args)
        if args.grouping:
            dataloader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=grouped_collate_fn, shuffle=False)
        else:
            dataloader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate_fn, shuffle=False)

        return dataloader, num_items, row_shapes, emb_shapes, None         

def gen_cache(sparse_data, cache_size, args):
    cached_item_idx = np.argsort(-np.array(sparse_data.sum(axis=0).reshape(-1))).reshape(-1)[:cache_size]
    sparse_cache = sparse_data[:, cached_item_idx[0]]
    sparse_data = sparse_data.tolil()
    sparse_data[:, cached_item_idx[0]] = 0

    for i in range(1, cache_size):
        sparse_cache = sp.hstack([sparse_cache, sparse_data[:, cached_item_idx[i]]])
        sparse_data[:, cached_item_idx[i]] = 0
    sparse_data = sparse_data.tocsr()

    sp.save_npz(f"/home/sminyu/rec_sys/saved_data/use_cache_{args.cache_size}_sparse_data.npz", sparse_data)
    sp.save_npz(f"/home/sminyu/rec_sys/saved_data/use_cache_{args.cache_size}_sparse_cache.npz", sparse_cache)

    logger.info("Finished Generating Cache Data")
    return sparse_cache.tocsr(), sparse_data

def count_cached_item_access(indices, num_items, args):
    _, acc_cnts = np.unique(indices, return_counts=True)
    
    sorted_acc_cnts = np.sort(acc_cnts)[::-1]
    total_acc = indices.shape[0]
    cached_num_items = int(num_items * args.cache_size // 100)

    num_cache_acc = sorted_acc_cnts[:cached_num_items].sum()

    per_cached_item_acc = np.round(num_cache_acc / total_acc, decimals=4) * 100

    print(f"Number of items : {num_items} in Dataset : {args.data_name}")
    print(f"Cache Size : {args.cache_size}, Percentage of cached item access : {per_cached_item_acc}")

def count_item_access(indices, num_items, args):
    _, acc_cnts = np.unique(indices, return_counts=True)
    num_acc_cnts, num_item_cnts = np.unique(acc_cnts, return_counts=True)

    total_acc = indices.shape[0]
    anu_idx = total_acc // num_items
    
    num_access = num_acc_cnts * num_item_cnts
    anu_access = (num_access[:3].sum() / num_access.sum()) * 100
    percent = np.round(anu_access, decimals=2)

    anu_items = (num_item_cnts[:3].sum() / num_items) * 100
    percent_item = np.round(anu_items, decimals=2)

    print(f"ANU : {anu_idx}, Number of access <= ANU : {percent} in Dataset : {args.data_name}")
    print(f"Number of Items <= ANU : {percent_item}")

def random_indices(indices, num_items):
    rand_arr = np.arange(num_items)
    np.random.shuffle(rand_arr)
    rand_indices = rand_arr[indices]

    return rand_indices

def sort_indices(indices, num_items):
    item_idx, cnts = np.unique(indices, return_counts=True)
    idx = np.argsort(-cnts)
    sort_arr = np.zeros([num_items], dtype=np.int64)
    sort_arr[idx] = np.arange(idx.size)
    sort_indices = sort_arr[indices]
    
    return sort_indices

def optimize_indices(indices, idx):

    optim_arr = np.zeros_like(idx)
    optim_arr[idx] = np.arange(idx.size)
    optim_indices = optim_arr[indices]
    
    return optim_indices

def collate_fn(datas):
    offsets = datas[0]["offsets"]
    indices = datas[0]["indices"]
    
    for data in datas[1:]:
        offset_idx = data["offsets"][1] - data["offsets"][0] + offsets[-1]
        offsets = torch.cat((offsets, torch.tensor([offset_idx], dtype=torch.long)))
        indices = torch.cat((indices, data["indices"]))

    return offsets, indices, None, None

def cached_collate_fn(datas):
    offsets = datas[0]["offsets"]
    indices = datas[0]["indices"]
    
    cached_offsets = datas[0]["cached_offsets"]
    cached_indices = datas[0]["cached_indices"]
    
    for data in datas[1:]:
        offset_idx = data["offsets"][1] - data["offsets"][0] + offsets[-1]
        offsets = torch.cat((offsets, torch.tensor([offset_idx], dtype=torch.long)))
        indices = torch.cat((indices, data["indices"]))
        
        cached_offset_idx = data["cached_offsets"][1] - data["cached_offsets"][0] + cached_offsets[-1]
        cached_offsets = torch.cat((cached_offsets, torch.tensor([cached_offset_idx], dtype=torch.long)))
        cached_indices = torch.cat((cached_indices, data["cached_indices"]))

    return offsets, indices, cached_offsets, cached_indices

def grouped_collate_fn(datas):

    first_offset = torch.tensor([0], dtype=torch.long)
        
    intra_group_offsets = torch.cat((first_offset, datas[0]["intra_group_offsets"]))
    intra_group_indices = datas[0]["intra_group_indices"]
    inter_group_offsets = datas[0]["inter_group_offsets"]
    inter_group_indices = datas[0]["inter_group_indices"]

    for data in datas[1:]:
        temp_offsets = data["intra_group_offsets"] + intra_group_offsets[-1]
        intra_group_offsets = torch.cat((intra_group_offsets, temp_offsets))
        intra_group_indices = torch.cat((intra_group_indices, data["intra_group_indices"]))
        
        inter_group_idx = data["inter_group_offsets"][1] - data["inter_group_offsets"][0] + inter_group_offsets[-1]
        inter_group_offsets = torch.cat((inter_group_offsets, torch.tensor([inter_group_idx], dtype=torch.long)))
        inter_group_indices = torch.cat((inter_group_indices, data["inter_group_indices"]))

    return intra_group_offsets, intra_group_indices, inter_group_offsets, inter_group_indices, None, None

def cached_grouped_collate_fn(datas):

    first_offset = torch.tensor([0], dtype=torch.long)
        
    intra_group_offsets = torch.cat((first_offset, datas[0]["intra_group_offsets"]))
    intra_group_indices = datas[0]["intra_group_indices"]
    inter_group_offsets = datas[0]["inter_group_offsets"]
    inter_group_indices = datas[0]["inter_group_indices"]

    cached_offsets = datas[0]["cached_offsets"]
    cached_indices = datas[0]["cached_indices"]
    
    for data in datas[1:]:
        temp_offsets = data["intra_group_offsets"] + intra_group_offsets[-1]
        intra_group_offsets = torch.cat((intra_group_offsets, temp_offsets))
        intra_group_indices = torch.cat((intra_group_indices, data["intra_group_indices"]))
        
        inter_group_idx = data["inter_group_offsets"][1] - data["inter_group_offsets"][0] + inter_group_offsets[-1]
        inter_group_offsets = torch.cat((inter_group_offsets, torch.tensor([inter_group_idx], dtype=torch.long)))
        inter_group_indices = torch.cat((inter_group_indices, data["inter_group_indices"]))
        
        cached_offset_idx = data["cached_offsets"][1] - data["cached_offsets"][0] + cached_offsets[-1]
        cached_offsets = torch.cat((cached_offsets, torch.tensor([cached_offset_idx], dtype=torch.long)))
        cached_indices = torch.cat((cached_indices, data["cached_indices"]))

    return intra_group_offsets, intra_group_indices, inter_group_offsets, inter_group_indices, cached_offsets, cached_indices

class RecDataset(Dataset):
    def __init__(self, offsets, indices, cache_offsets, cache_indices, args):
        self.offsets = offsets
        self.indices = indices
        self.use_cache = False
        
        if args.use_cache:
            self.cache_offsets = cache_offsets
            self.cache_indices = cache_indices
            self.use_cache = True
            
        self.grouping = False
        
        if args.grouping:
            self.grouping = True
            self.group_size = args.group_size
        
    def __len__(self):
        return len(self.offsets) - 1
    
    def __getitem__(self, idx):
        
        indices = self.indices[self.offsets[idx] : self.offsets[idx+1]]
        offsets = torch.tensor([0, len(indices)])

        instance = dict()
        if self.use_cache:
            cached_indices = self.cache_indices[self.offsets[idx] : self. offsets[idx+1]]
            cached_offsets = torch.tensor([0, len(cached_indices)])
        
            instance["cached_offsets"] = cached_offsets
            instance["cached_indices"] = cached_indices
        
        if self.grouping:
            group_idx = indices // self.group_size
            
            inter_group_indices, cnts = group_idx.unique(return_counts=True)
            inter_group_offsets = torch.tensor([0, len(inter_group_indices)])
            
            intra_group_offsets = cnts.cumsum(dim=0)
            intra_group_indices = indices % self.group_size

            instance["inter_group_offsets"] = inter_group_offsets
            instance["inter_group_indices"] = inter_group_indices
            instance["intra_group_offsets"] = intra_group_offsets
            instance["intra_group_indices"] = intra_group_indices
        
        instance["offsets"] = offsets
        instance["indices"] = indices
        
        return instance
