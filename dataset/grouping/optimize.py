import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
import time
from tqdm import tqdm

def optim_indices(data, group_size, args):
    
    num_users, num_items = data.shape[0], data.shape[1]
    optim_idx_dict = dict()
    optim_idx_dict["residue"] = np.array([], dtype=np.int32)
    target_freq_list = [1, 2, 3, 4]
    
    num_freq_list = np.array(data.sum(axis=0)).squeeze()
    
    for target_freq in target_freq_list:
        target_item_idx = select_num_freq_items(num_freq_list, target_freq)
        optim_idx = fit(data, target_item_idx, target_freq)

        residue_num_items = optim_idx.shape[0] % group_size
        optim_idx_dict[f"{target_freq}"] = optim_idx[:-residue_num_items]
        
        if "residue" in optim_idx_dict.keys():
            optim_idx_dict["residue"] = np.append(optim_idx_dict["residue"], optim_idx[-residue_num_items:])
        else:
            optim_idx_dict["residue"] = optim_idx[-residue_num_items:]

        import pdb; pdb.set_trace()

    remain_idx = select_remain_items(num_freq_list, remain_freq=target_freq_list[-1])
    sorted_remain_idx = np.argsort(num_freq_list[remain_idx])[::-1]
    rest_idx = remain_idx[sorted_remain_idx]
    
    residue_num_items = rest_idx.shape[0] % group_size
    optim_idx_dict["rest"] = rest_idx[:-residue_num_items]
    optim_idx_dict["residue"] = np.append(optim_idx_dict["residue"], rest_idx[-residue_num_items:])
    
    optim_indices = create_optimized_indices(optim_idx_dict, target_freq_list)
    np.save(f"optim_indices/{args.data_name}", optim_indices)

def create_optimized_indices(optim_idx_dict, freq_list):
    optimized_indices = optim_idx_dict["rest"]
    num = freq_list[-1] + 1
    
    for freq in freq_list:
        optimized_indices = np.append(optimized_indices, optim_idx_dict[f"{num - freq}"])

    optimized_indices = np.append(optimized_indices, optim_idx_dict["residue"])
        
    return optimized_indices
    
def fit(data, target_item_idx, target_freq):
    
    optim_idx = np.array([], dtype=np.int32)
    target_data = data[:, target_item_idx]
    target_data = sort_by_interacts(target_data, target_freq)
    
    if target_freq == 1:
        optim_idx = np.append(optim_idx, target_item_idx[target_data.indices])
        return optim_idx
    
    indices = target_data.indices
    offsets = target_data.indptr
    num_iter = offsets.shape[0] - 1

    for i in tqdm(range(num_iter)):
        hier_item_idx = indices[offsets[i]:offsets[i + 1]]
        hier_optim_idx = fit(target_data, hier_item_idx, target_freq - 1)
        optim_idx = np.append(optim_idx, hier_optim_idx)
        
    return optim_idx
        
def sort_by_interacts(target_data, target_freq):
    
    num_interact_list = np.array(target_data.sum(axis=1)).squeeze()
    if target_freq > 1:
        target_user_idx = np.argsort(num_interact_list)[::-1][1:]
    else:
        target_user_idx = np.argsort(num_interact_list)[::-1]

    target_data = target_data[target_user_idx]
    nnz_user_idx = np.where(target_data.sum(axis=1)!=0)[0]
    target_data = target_data[nnz_user_idx]
    
    return target_data

def select_remain_items(num_freq_list, remain_freq):
    return np.where(num_freq_list>remain_freq)[0]

def select_num_freq_items(num_freq_list, target_freq):
    return np.where(num_freq_list==target_freq)[0]