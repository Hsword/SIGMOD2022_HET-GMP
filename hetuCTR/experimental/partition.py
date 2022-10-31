import numpy as np
import scipy.sparse as sp
import time
import argparse
import os.path as osp

import hetuCTR_partition

def load_criteo_data():
    path = osp.dirname(__file__)
    fname = osp.join(path, "train_sparse_feats.npy")
    assert osp.exists(fname)
    data = np.load(fname)
    if not data.data.c_contiguous:
        data = np.ascontiguousarray(data)
    return data

def get_comm_mat(nrank, ngpus):
    mat = np.ones([nrank, nrank])
    for i in range(nrank):
        for j in range(nrank):
            if i == j:
                mat[i, j] = 0
            elif i // ngpus == j // ngpus:
                mat[i, j] = 0.1
            else:
                mat[i, j] = 1
    return mat

def direct_partition(data, nparts, ngpus, batch_size, rerun):
    start = time.time()
    mat = get_comm_mat(nparts, ngpus)
    partition = hetuCTR_partition.partition(data, mat, nparts, batch_size)

    cost = partition.get_communication()
    print("Initial cost : {}".format(np.multiply(cost, mat).sum()))
    for i in range(rerun):
        partition.refine_data()
        partition.refine_embed()
        cost = partition.get_communication()
        print("Refine round {} : {}".format(i, np.multiply(cost, mat).sum()))
        print("Data :", partition.get_data_cnt())
        print("Embed:", partition.get_embed_cnt())
        print("In   :", np.sum(cost, axis=0))
        print("Out  :", np.sum(cost, axis=1))
        print(cost.astype(np.int))
    item_partition, idx_partition = partition.get_result()
    print("Partition Time : ", time.time()-start)
    start = time.time()

    arr_dict = {"embed_partition" : idx_partition, "data_partition" : item_partition}
    priority = partition.get_priority()
    for i in range(nparts):
        idxs = np.where(idx_partition==i)[0]
        priority[i][idxs] = -1 # remove embedding that has been stored
        arr = np.argsort(priority[i])[len(idxs):][ : : -1]
        arr_dict[str(i)] = arr
    print("Sort priority Time : ", time.time()-start)

    np.savez(args.output, **arr_dict)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--nrank", "-n" , type=int, default=8)
    parser.add_argument("--batch_size", "-b" , type=int, default=8192)
    parser.add_argument("--rerun", "-r" , type=int, default=10)
    parser.add_argument("--ngpus", "-g" , type=int, default=8)
    parser.add_argument("--output", "-o" , type=str, default="partition.npz")
    args = parser.parse_args()
    data = load_criteo_data()
    direct_partition(data, args.nrank, args.ngpus, args.batch_size, args.rerun)
