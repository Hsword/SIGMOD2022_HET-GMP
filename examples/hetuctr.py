import hetu as ht
from hetu.launcher import launch
from hetu import init
from  hetu.gpu_ops.SharedTable import SharedTableOp
from hetu.communicator.mpi_nccl_comm import ncclDataType_t, ncclRedOp_t

from models.hetuctr_data import load_dataset
from models.hetuctr_models import WDL, DCN, DFM

import os.path as osp
import numpy as np
import yaml
import time
import argparse
from sklearn import metrics

def comm_sync_data(comm, *args):
    array = ht.array(args, ht.cpu())
    comm.dlarrayNcclAllReduce(array, array, ncclDataType_t.ncclFloat32, ncclRedOp_t.ncclSum, comm.stream)
    comm.stream.sync()
    return array.asnumpy() / comm.nRanks.value

def worker(args):
    def train(iterations, auc_enabled=False):
        localiter = range(iterations)
        train_loss = []
        train_acc = []
        if auc_enabled:
            train_auc = []
        for it in localiter:
            loss_val, predict_y, y_val, _ = executor.run('train', convert_to_numpy_ret_vals=True)
            if y_val.shape[1] == 1: # for criteo case
                acc_val = np.equal(
                    y_val,
                    predict_y > 0.5).astype(float)
            else:
                acc_val = np.equal(
                    np.argmax(y_val, 1),
                    np.argmax(predict_y, 1)).astype(float)
            train_loss.append(loss_val[0])
            train_acc.append(acc_val)
            if auc_enabled:
                train_auc.append(metrics.roc_auc_score(y_val, predict_y))
        if auc_enabled:
            return np.mean(train_loss), np.mean(train_acc), np.mean(train_auc)
        else:
            return np.mean(train_loss), np.mean(train_acc), 0
    def validate(iterations):
        localiter = range(iterations)
        test_loss = []
        test_acc = []
        test_auc = []
        for it in localiter:
            loss_val, test_y_predicted, y_test_val = executor.run('validate', convert_to_numpy_ret_vals=True)
            if y_test_val.shape[1] == 1: # for criteo case
                correct_prediction = np.equal(
                    y_test_val,
                    test_y_predicted > 0.5).astype(float)
            else:
                correct_prediction = np.equal(
                    np.argmax(y_test_val, 1),
                    np.argmax(test_y_predicted, 1)).astype(float)
            test_loss.append(loss_val[0])
            test_acc.append(correct_prediction)
            test_auc.append(metrics.roc_auc_score(y_test_val, test_y_predicted))
        return np.mean(test_loss), np.mean(test_acc), np.mean(test_auc)

    def get_shard(data):
        part_size = data.shape[0] // nrank
        start = part_size * rank
        end = start + part_size if rank != nrank - 1 else data.shape[0]
        return data[start:end]

    def get_partitioned_shard(data):
        if data_arr is not None:
            return data[np.where(data_arr==rank)]
        else:
            return get_shard(data)

    batch_size = args.batch_size

    nrank = comm.nRanks.value

    dense, sparse, labels = load_dataset(args.dataset, val=False)
    has_dense_feature = dense is not None
    dense_input = [[get_partitioned_shard(dense), batch_size, 'train']] if has_dense_feature else None
    sparse_input = [ht.Dataloader(get_partitioned_shard(sparse).astype(np.int64), batch_size, 'train', use_numpy=True)]
    y_ = [[get_partitioned_shard(labels), batch_size, 'train']]

    if args.val:
        val_dense, val_sparse, val_labels = load_dataset(args.dataset, val=True)
        if has_dense_feature:
            dense_input.append([get_shard(val_dense), batch_size, 'validate'])
        sparse_input.append(ht.Dataloader(get_shard(val_sparse).astype(np.int64), batch_size, 'validate', use_numpy=True))
        y_.append([get_shard(val_labels), batch_size, 'validate'])

    dense_input = ht.dataloader_op(dense_input) if has_dense_feature else None
    sparse_input = ht.dataloader_op(sparse_input)
    y_ = ht.dataloader_op(y_)

    print("Data loaded.")
    models = {"wdl" : WDL, "dcn" : DCN, "dfm" : DFM}
    loss, prediction, y_, train_op = models[args.model](args.dataset, dense_input, sparse_input, y_, args.embed_dim, rank, nrank, device_id,
        args.bound, root_arr, storage_arr)

    eval_nodes = {'train': [loss, prediction, y_, train_op]}
    if args.val:
        print('Validation enabled...')
        eval_nodes['validate'] = [loss, prediction, y_]
    executor = ht.Executor(eval_nodes, ctx=ht.gpu(device_id), comm_mode="AllReduce" if nrank > 1 else None, seed=123, log_path='./logs/')

    if rank == 0:
        log_file = open(args.output, 'w')
    for ep in range(args.iter // args.log_every):
        ep_st = time.time()
        train_loss, train_acc, train_auc = train(args.log_every)
        ep_en = time.time()
        train_time, train_loss, train_acc, train_auc = comm_sync_data(comm, ep_en - ep_st, train_loss, train_acc, train_auc)
        if rank==0:
            printstr = "TRAIN %d: loss %.4f acc %.4f time %.4f speed %d" % (ep * args.log_every, train_loss, train_acc, train_time, args.log_every*batch_size/train_time)
            print(printstr, flush=True)
            print(printstr, file=log_file, flush=True)
        if args.val and ep > 0 and ep % (args.eval_every // args.log_every) == 0:
            val_loss, val_acc, val_auc = validate(executor.get_batch_num('validate'))
            val_loss, val_acc, val_auc = comm_sync_data(comm, val_loss, val_acc, val_auc)
            if rank==0:
                printstr = "EVAL %d: val_loss %.4f val_acc %.4f val_auc %.4f" % (ep * args.log_every, val_loss, val_acc, val_auc)
                print(printstr, flush=True)
                print(printstr, file=log_file, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="[criteo avazu company]")
    parser.add_argument("--model", type=str, required=True, help="[wdl dcn dfm]")
    parser.add_argument("--val", action="store_true", help="whether to use validation")
    parser.add_argument("--bound", type=int, default=10, help="cache bound")
    parser.add_argument("--batch_size", type=int, default=128, help="batch size")
    parser.add_argument("--embed_dim", type=int, default=128, help="embedding dim")
    parser.add_argument("--iter", type=int, default=10000, help="nnumber of iteration")
    parser.add_argument("--log_every", type=int, default=200)
    parser.add_argument("--eval_every", type=int, default=10000)
    parser.add_argument("--store_rate", type=float, default=0.01)
    parser.add_argument("--partition", type=str, default=None)
    parser.add_argument("--output", type=str, default="hetuctr.log")
    args = parser.parse_args()

    comm, device_id = ht.mpi_nccl_init()
    rank = comm.myRank.value
    if args.partition:
        args.partition = osp.normpath(osp.expanduser(args.partition))
        assert osp.exists(args.partition)
        partition = np.load(args.partition)
        data_arr = partition["data_partition"]
        root_arr = partition["embed_partition"]
        storage_arr = partition[str(rank)]
        storage_arr = storage_arr[:int(args.store_rate * len(storage_arr))]
        storage_arr = np.concatenate([np.where(root_arr==rank)[0], storage_arr])
    else:
        data_arr, root_arr, storage_arr = None, None, None

    worker(args)
    ht.mpi_nccl_finish(comm)
