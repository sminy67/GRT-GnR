import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
import time
from tqdm import tqdm

def optim_indices(indices, data, group_size, args):
    
    num_users, num_items = data.shape[0], data.shape[1]
    
    print(f"Number of Users : {num_users}, Number of Items : {num_items} in {args.data_name} Dataset !")
    
    optimized_indices = np.array([], dtype=np.int32)
    residue_indices = np.array([], dtype=np.int32)
    
    # Number of users interacting each item
    freq_item_list = np.array(data.sum(axis=0)).squeeze()
    
    freq_list = [1, 2, 3, 4]
    for freq in freq_list:
        # Chooses the item index of target frequency from total data (e.g. |u_i| = 1, 2, 3...)
        freq_item_idx = select_freq_items(freq_item_list, freq)
        opt_idx, res_idx = fit(data, freq_item_idx, freq, group_size)

        optimized_indices = np.append(optimized_indices, opt_idx)
        residue_indices = np.append(residue_indices, res_idx)
    
    optimized_indices = np.append(optimized_indices, residue_indices)
    remain_idx = select_remain_items(freq_item_list, freq_list[-1])
    sorted_remain_idx = np.argsort(freq_item_list[remain_idx])[::-1]
    rest_idx = remain_idx[sorted_remain_idx]
    optimized_indices = np.append(rest_idx, optimized_indices)
    np.save(f"/home/sminyu/rec_sys/GRT-GnR/dataset/grouping/optim_data/{args.data_name}", optimized_indices)

    arr = np.zeros_like(optimized_indices)
    arr[optimized_indices] = np.arange(optimized_indices.size)
    optim_indices = arr[indices]

    return optim_indices
    
'''
def fit(data, item_idx, freq, group_size, is_hier=False):
    
    opt_idx = np.array([], dtype=np.int32)
    tmp_idx = np.array([], dtype=np.int32)
    res_idx = np.array([], dtype=np.int32)
    remain_item_idx = np.copy(item_idx)
    
    # Data that have the target frequency items (i.e. every item has equal number of user interaction)
    freq_data = data[:, item_idx]
    
    # Eliminate the user who does not interact with target items
    freq_data = remove_zero_users(freq_data)
    
    # Sort data in descending order based on user that has most interaction with items
    freq_data = sort_freq_data(freq_data, is_hier)
    
    indices = freq_data.indices
    offsets = freq_data.indptr
    
    if (freq == 1):
        for i in tqdm(range(len(offsets) - 1)):
            start = offsets[i]
            end = offsets[i+1]
            
            grp_end = (end - start) // group_size
            
            if (grp_end == 0):
                tmp_idx = np.append(opt_idx, item_idx[indices[start : end]])
            else:
                opt_idx = np.append(opt_idx, item_idx[indices[start : group_size * grp_end]])
                res_idx = np.append(res_idx, item_idx[indices[group_size * grp_end : end]])
                
        return opt_idx, tmp_idx, res_idx
    else:
        for i in range(len(offsets) - 1):
            start = offsets[i]
            end = offsets[i+1]
            
            cur_item_idx = indices[offsets[i] : offsets[i+1]]
            if (i > 0):
                prev_item_idx = indices[offsets[i-1] : offsets[i]]
                remain_item_idx = np.setdiff1d(remain_item_idx, prev_item_idx)
                if len(remain_item_idx) == 0:
                    break
                
                cur_item_idx = np.intersect1d(cur_item_idx, remain_item_idx)
                if len(cur_item_idx) == 0:
                    continue
            hier_opt_idx, hier_tmp_idx, hier_res_idx = fit(freq_data, cur_item_idx, freq - 1, group_size, is_hier=True)
            opt_idx = np.append(opt_idx, hier_opt_idx)
            tmp_idx = np.append(tmp_idx, hier_tmp_idx)
            res_idx = np.append(res_idx, hier_res_idx)
        return opt_idx, tmp_idx, res_idx
'''
def fit(data, item_idx, freq, group_size, is_hier=False):
    
    opt_idx = np.array([], dtype=np.int32)
    res_idx = np.array([], dtype=np.int32)
    
    # Data that have the target frequency items (i.e. every item has equal number of user interaction)
    freq_data = data[:, item_idx]
    
    # Eliminate the user who does not interact with target items
    freq_data = remove_zero_users(freq_data)
    
    # Sort data in descending order based on user that has most interaction with items
    freq_data = sort_freq_data(freq_data, is_hier)
    
    remain_item_idx = np.arange(freq_data.shape[1])
    indices = freq_data.indices
    offsets = freq_data.indptr
    
    if (freq == 1):
        split_point = len(indices) % group_size
        return item_idx[indices[:-split_point]], item_idx[indices[-split_point:]]
    else:
        for i in range(len(offsets) - 1):
            start = offsets[i]
            end = offsets[i+1]
            
            cur_item_idx = indices[start : end]
            if (i > 0):
                cur_item_idx = np.intersect1d(cur_item_idx, remain_item_idx)
                if len(cur_item_idx) == 0:
                    continue
            hier_opt_idx, hier_res_idx = fit(freq_data, cur_item_idx, freq - 1, group_size, is_hier=True)

            opt_idx = np.append(opt_idx, hier_opt_idx)
            res_idx = np.append(res_idx, hier_res_idx)

            remain_item_idx = np.setdiff1d(remain_item_idx, cur_item_idx)
            if len(remain_item_idx) == 0:
                break
        return item_idx[opt_idx], item_idx[res_idx]

def select_remain_items(freq_item_list, freq):
    return np.where(freq_item_list>freq)[0]

def sort_freq_data(data, is_hier=False):
    
    interact_user_list = np.array(data.sum(axis=1)).squeeze()
    if is_hier:
        user_idx = np.argsort(interact_user_list)[::-1][1:]
    else:
        user_idx = np.argsort(interact_user_list)[::-1]
    return data[user_idx]

def remove_zero_users(data):
    nnz_user_idx = np.where(data.sum(axis=1)!=0)[0]
    return data[nnz_user_idx]    
    
def select_freq_items(freq_item_list, freq):
    return np.where(freq_item_list==freq)[0]