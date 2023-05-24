import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
import time
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)

def optim_indices(indices, data, group_size, args):
    
    logger.info(f"Start Optimizing Items in {args.data_name} Dataset")
    
    # Number of users interacting each item
    item_freq = check_freq(data)
    
    target_freq = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

    not_optimized_idx = select_remain_items(item_freq, target_freq[-1])
    sorted_remain_idx = sort_data_with_items(item_freq, not_optimized_idx)
    
    grouped_indices, residue_indices = split_group_index(sorted_remain_idx, group_size)
    
    for freq in target_freq:
        # Chooses the item index of target frequency from total data (e.g. [u_i] = 1, 2, 3 ...)
        target_item_idx = select_items(item_freq, freq)
        
        logger.info(f"Target Frequency {freq} & Number of Items : {target_item_idx.shape[0]}")
        
        # Optimizing the items in the target frequency
        opt_item_idx, res_item_idx = fit(freq, data, target_item_idx, group_size)
        grouped_indices = np.append(grouped_indices, opt_item_idx)
        residue_indices = np.append(residue_indices, res_item_idx)

    output = concate_output(grouped_indices, residue_indices)
    
    logger.info(f"Finished Optimizing Items in {args.data_name} Dataset & Saving")
    
    np.save(f"/home/sminyu/rec_sys/GRT-GnR/dataset/grouping/optim_data/{args.data_name}", output)
    
    output = make_indices(indices, output)

    return output

def make_indices(indices, idx):
    
    arr = np.zeros_like(idx)
    arr[idx] = np.arange(idx.size)
    output = arr[indices]

    return output

def fit(freq, data, item_idx, group_size):
    
    grouped_idx = np.array([], dtype=np.int32)
    residue_idx = np.array([], dtype=np.int32)
    
    # Data that have the target frequency items (i.e. every item has equal number of user interaction)
    target_data = data[:, item_idx]
    
    # Eliminating the users who do not interact with target items
    target_data = remove_users(target_data)
    
    # Sorting data in descending order based on user that has most interaction with items
    target_data = sort_data_with_user(target_data)

    remain_idx = np.arange(target_data.shape[1])
    indices = target_data.indices
    offsets = target_data.indptr
    
    if (freq == 1):
        grouped_idx, residue_idx = split_group_index(indices, group_size)

        return item_idx[grouped_idx], item_idx[residue_idx]
    else:
        for i in tqdm(range(len(offsets) - 1)):
            start = offsets[i]
            end = offsets[i + 1]
            
            cur_idx = indices[start : end]
            if (i > 0):
                cur_idx = np.intersect1d(cur_idx, remain_idx)
                if len(cur_idx) == 0:
                    continue
            hier_opt_indices = hier_fit(target_data, cur_idx, freq - 1)
            hier_grouped_idx, hier_residue_idx = split_group_index(hier_opt_indices, group_size)
            grouped_idx = np.append(grouped_idx, hier_grouped_idx)
            residue_idx = np.append(residue_idx, hier_residue_idx)

            remain_idx = np.setdiff1d(remain_idx, cur_idx)
            if len(remain_idx) == 0:
                break

        return item_idx[grouped_idx], item_idx[residue_idx]
            
def hier_fit(data, item_idx, freq):
    
    hier_idx = np.array([], dtype=np.int32)
    
    # Data that have the target frequency items (i.e. every item has equal number of user interaction)
    target_data = data[:, item_idx]
    
    # Eliminating the users who do not interact with target items
    target_data = remove_users(target_data)
    
    # Sorting data in descending order based on user that has most interaction with items
    target_data = sort_data_with_user(target_data, is_hier=True)
    
    remain_idx = np.arange(target_data.shape[1])
    indices = target_data.indices
    offsets = target_data.indptr
    
    if (freq == 1):
        hier_idx = indices
        return item_idx[hier_idx]
    else:
        for i in range(len(offsets) - 1):
            start = offsets[i]
            end = offsets[i + 1]
            
            cur_idx = indices[start : end]
            if (i > 0):
                cur_idx = np.intersect1d(cur_idx, remain_idx)
                if len(cur_idx) == 0:
                    continue
            recur_idx = hier_fit(target_data, cur_idx, freq - 1)
            hier_idx = np.append(hier_idx, recur_idx)
            
            remain_idx = np.setdiff1d(remain_idx, cur_idx)
            if len(remain_idx) == 0:
                break
            
        return item_idx[hier_idx]

def concate_output(grp_idx, res_idx):
    output = np.array([], dtype=np.int32)
    
    output = np.append(output, grp_idx)
    output = np.append(output, res_idx)
    return output
                
def split_group_index(indices, group_size):
    split_idx = len(indices) % group_size
    return indices[:-split_idx], indices[-split_idx:]            

def sort_data_with_items(item_freq, idx):
    sorted_idx = np.argsort(item_freq[idx])[::-1]
    return idx[sorted_idx]

def sort_data_with_user(data, is_hier=False):
    num_interact = np.array(data.sum(axis=1)).squeeze()
    if is_hier:
        user_idx = np.argsort(num_interact)[::-1][1:]
    else:
        user_idx = np.argsort(num_interact)[::-1]
    return data[user_idx]
   
def remove_users(data):
    user_idx = np.where(data.sum(axis=1)!=0)[0]
    return data[user_idx]

def select_remain_items(item_freq, freq):
    return np.where(item_freq>freq)[0]

def select_items(item_freq, freq):
    return np.where(item_freq==freq)[0]    

def check_freq(data):
    return np.array(data.sum(axis=0)).squeeze()