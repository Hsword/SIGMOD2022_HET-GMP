import os.path as osp
import numpy as np
dataset_root = osp.expanduser("~/hetuctr_dataset")

def split_dataset(name):
    assert name in ("criteo", "avazu", "company")
    root = osp.join(dataset_root, name)
    assert osp.isdir(root)
    feat_all = np.load(osp.join(root, "sparse.npy"))
    label_all = np.load(osp.join(root, "labels.npy"))
    n = len(feat_all)
    idx = np.random.permutation(n)
    test_idx, train_idx = idx[:int(n * 0.1)], idx[int(n * 0.1):]
    feat_train, label_train = feat_all[train_idx], label_all[train_idx]
    feat_test, label_test = feat_all[test_idx], label_all[test_idx]
    np.save(osp.join(root, "train_sparse_feats.npy"), feat_train)
    np.save(osp.join(root, "test_sparse_feats.npy"), feat_test)
    np.save(osp.join(root, "train_labels.npy"), label_train)
    np.save(osp.join(root, "test_labels.npy"), label_test)

def load_dataset(name, val=False):
    assert name in ("criteo", "avazu", "company")
    root = osp.join(dataset_root, name)
    assert osp.isdir(root)
    if not val:
        sparse = np.load(osp.join(root, "train_sparse_feats.npy"))
        label = np.load(osp.join(root, "train_labels.npy"))
        if name == "criteo":
            dense = np.load(osp.join(root, "train_dense_feats.npy"))
        else:
            dense = None
    else:
        sparse = np.load(osp.join(root, "test_sparse_feats.npy"))
        label = np.load(osp.join(root, "test_labels.npy"))
        if name == "criteo":
            dense = np.load(osp.join(root, "test_dense_feats.npy"))
        else:
            dense = None
    return dense, sparse, label
