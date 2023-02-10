import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.sparse import coo_matrix, csr_matrix

def load_data(args):
    
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

    row = data.user_id.values
    col = data.item_id.values
    val = data.rating.values
    
    sparse_data = coo_matrix((val, (row, col)), shape=(num_users, num_items)).tocsr()
    indices = sparse_data.indices
    offsets = sparse_data.indptr

    dataset = RecDataset(torch.tensor(offsets, dtype=torch.long), torch.tensor(indices, dtype=torch.long), args)

    if args.grouping:
        dataloader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=grouped_collate_fn, shuffle=False)
    else:
        dataloader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate_fn, shuffle=False)
        
    return dataloader, num_items

def collate_fn(datas):
    offsets = datas[0]["offsets"]
    indices = datas[0]["indices"]
    
    for data in datas[1:]:
        offset_idx = data["offsets"][1] - data["offsets"][0] + offsets[-1]
        offsets = torch.cat((offsets, torch.tensor([offset_idx], dtype=torch.long)))
        indices = torch.cat((indices, data["indices"]))

    return offsets, indices 

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

    return intra_group_offsets, intra_group_indices, inter_group_offsets, inter_group_indices    

class RecDataset(Dataset):
    def __init__(self, offsets, indices, args):
        self.offsets = offsets
        self.indices = indices
        self.grouping = False
        
        if args.grouping:
            self.grouping = True
            self.group_size = args.group_size
        
    def __len__(self):
        return len(self.offsets)
    
    def __getitem__(self, idx):

        offsets = self.offsets[idx : idx + 2]
        indices = self.indices[self.offsets[idx] : self.offsets[idx+1]]

        instance = dict()
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