import hetu as ht
from hetu import init
from  hetu.gpu_ops.SharedTable import SharedTableOp

dataset_sparse_dimension = {"avazu" : 9449445, "company" : 66102027, "criteo" : 33762577}
dataset_sparse_slots = {"avazu" : 22, "company" : 43, "criteo" : 26}
dataset_dense_dim = {"avazu" : 0, "company" : 0, "criteo" : 13}

def WDL(name, dense_input, sparse_input, y_, embed_dim , rank, nrank, device_id, bound, root_arr, storage_arr):
    feature_dimension = dataset_sparse_dimension[name]
    slots = dataset_sparse_slots[name]
    dense_dim = dataset_dense_dim[name]
    learning_rate = 0.01

    if nrank == 1:
        Embedding = init.random_normal([feature_dimension, embed_dim], stddev=0.01, ctx=ht.gpu(device_id))
        sparse_input = ht.embedding_lookup_op(Embedding, sparse_input, ctx=ht.gpu(device_id))
    else:
        sparse_input = SharedTableOp(
            sparse_input, rank, nrank, (feature_dimension, embed_dim), learning_rate, bounds=(bound, bound),
            root_arr=root_arr, storage_arr=storage_arr, ctx=ht.gpu(device_id))
    sparse_input = ht.array_reshape_op(sparse_input, (-1, slots * embed_dim))
    if dense_input is not None:
        #DNN
        flatten = dense_input
        W1 = init.random_normal([dense_dim, 256], stddev=0.01, name = "W1")
        W2 = init.random_normal([256, 256], stddev=0.01, name = "W2")
        W3 = init.random_normal([256, 256], stddev=0.01, name = "W3")

        W4 = init.random_normal([256 + slots * embed_dim, 1], stddev=0.01, name = "W4")

        fc1 = ht.matmul_op(flatten, W1)
        relu1 = ht.relu_op(fc1)
        fc2 = ht.matmul_op(relu1, W2)
        relu2 = ht.relu_op(fc2)
        y3 = ht.matmul_op(relu2, W3)

        y4 = ht.concat_op(sparse_input, y3, axis = 1)
    else:
        W4 = init.random_normal([slots * embed_dim, 1], stddev=0.01, name = "W4")
        y4 = sparse_input

    y = ht.matmul_op(y4, W4)
    y = ht.sigmoid_op(y)

    loss = ht.binarycrossentropy_op(y, y_)
    loss = ht.reduce_mean_op(loss, [0])
    opt = ht.optim.SGDOptimizer(learning_rate=learning_rate)
    train_op = opt.minimize(loss)

    return loss, y, y_, train_op

def DCN(name, dense_input, sparse_input, y_, embed_dim , rank, nrank, device_id, bound, root_arr, storage_arr):
    feature_dimension = dataset_sparse_dimension[name]
    slots = dataset_sparse_slots[name]
    dense_dim = dataset_dense_dim[name]
    learning_rate = 0.003

    if nrank == 1:
        Embedding = init.random_normal([feature_dimension, embed_dim], stddev=0.01, ctx=ht.gpu(device_id))
        sparse_input = ht.embedding_lookup_op(Embedding, sparse_input, ctx=ht.gpu(device_id))
    else:
        sparse_input = SharedTableOp(
            sparse_input, rank, nrank, (feature_dimension, embed_dim), learning_rate, bounds=(bound, bound),
            root_arr=root_arr, storage_arr=storage_arr, ctx=ht.gpu(device_id))
    sparse_input = ht.array_reshape_op(sparse_input, (-1, slots * embed_dim))
    if dense_input is not None:
        x = ht.concat_op(sparse_input, dense_input, axis=1)
    else:
        x = sparse_input

    def cross_layer(x0, x1):
        embedding_len = slots * embed_dim + dense_dim
        weight = init.random_normal(shape=(embedding_len, 1), stddev=0.01, name='weight')
        bias = init.random_normal(shape=(embedding_len,), stddev=0.01, name='bias')
        x1w = ht.matmul_op(x1, weight) #(batch_size, 1)
        y = ht.mul_op(x0, ht.broadcastto_op(x1w, x0))
        y = y + x1 + ht.broadcastto_op(bias, y)
        return y

    def build_cross_layer(x0, num_layers = 3):
        x1 = x0
        for i in range(num_layers):
            x1 = cross_layer(x0, x1)
        return x1
    # Cross Network
    cross_output = build_cross_layer(x, num_layers = 1)

    #DNN
    flatten = x
    W1 = init.random_normal([slots * embed_dim + dense_dim, 256], stddev=0.01, name = "W1")
    W2 = init.random_normal([256, 256], stddev=0.01, name = "W2")
    W3 = init.random_normal([256, 256], stddev=0.01, name = "W3")

    W4 = init.random_normal([256 + slots * embed_dim + dense_dim, 1], stddev=0.01, name = "W4")

    fc1 = ht.matmul_op(flatten, W1)
    relu1 = ht.relu_op(fc1)
    fc2 = ht.matmul_op(relu1, W2)
    relu2 = ht.relu_op(fc2)
    y3 = ht.matmul_op(relu2, W3)

    y4 = ht.concat_op(cross_output, y3, axis = 1)
    y = ht.matmul_op(y4, W4)
    y = ht.sigmoid_op(y)

    loss = ht.binarycrossentropy_op(y, y_)
    loss = ht.reduce_mean_op(loss, [0])
    opt = ht.optim.SGDOptimizer(learning_rate=learning_rate)
    train_op = opt.minimize(loss)

    return loss, y, y_, train_op

def DFM(name, dense_input, sparse_input, y_, embed_dim , rank, nrank, device_id, bound, root_arr, storage_arr):
    feature_dimension = dataset_sparse_dimension[name]
    slots = dataset_sparse_slots[name]
    dense_dim = dataset_dense_dim[name]
    learning_rate = 0.001

    # FM
    if nrank == 1:
        Embedding1 = init.random_normal([feature_dimension, 1], stddev=0.01, ctx=ht.gpu(device_id))
        Embedding2 = init.random_normal([feature_dimension, embed_dim], stddev=0.01, ctx=ht.gpu(device_id))
        sparse_1dim_input = ht.embedding_lookup_op(Embedding1, sparse_input, ctx=ht.gpu(device_id))
        sparse_2dim_input = ht.embedding_lookup_op(Embedding2, sparse_input, ctx=ht.gpu(device_id))
    else:
        sparse_input = SharedTableOp(
            sparse_input, rank, nrank, (feature_dimension, embed_dim + 1), learning_rate, bounds=(bound, bound),
            root_arr=root_arr, storage_arr=storage_arr, ctx=ht.gpu(device_id))

        sparse_1dim_input = ht.slice_op(sparse_input, (0, 0, 0), (-1, -1, 1))
        sparse_2dim_input = ht.slice_op(sparse_input, (0, 0, 1), (-1, -1, embed_dim))

    fm_sparse_part = ht.reduce_sum_op(sparse_1dim_input, axes = 1)
    if dense_input is not None:
        FM_W = init.random_normal([dense_dim, 1], stddev=0.01, name="dense_parameter")
        fm_dense_part = ht.matmul_op(dense_input, FM_W)
        y1 = fm_dense_part + fm_sparse_part
    else:
        y1 = fm_sparse_part

    sparse_2dim_sum = ht.reduce_sum_op(sparse_2dim_input, axes = 1)
    sparse_2dim_sum_square = ht.mul_op(sparse_2dim_sum, sparse_2dim_sum)

    sparse_2dim_square = ht.mul_op(sparse_2dim_input, sparse_2dim_input)
    sparse_2dim_square_sum = ht.reduce_sum_op(sparse_2dim_square, axes = 1)
    sparse_2dim = sparse_2dim_sum_square +  -1 * sparse_2dim_square_sum
    sparse_2dim_half = sparse_2dim * 0.5
    """snd order output"""
    y2 = ht.reduce_sum_op(sparse_2dim_half, axes = 1, keepdims = True)

    #DNN
    flatten = ht.array_reshape_op(sparse_2dim_input,(-1, slots * embed_dim))
    W1 = init.random_normal([slots * embed_dim, 256], stddev=0.01, name = "W1")
    W2 = init.random_normal([256, 256], stddev=0.01, name = "W2")
    W3 = init.random_normal([256, 1], stddev=0.01, name = "W3")

    fc1 = ht.matmul_op(flatten, W1)
    relu1 = ht.relu_op(fc1)
    fc2 = ht.matmul_op(relu1, W2)
    relu2 = ht.relu_op(fc2)
    y3 = ht.matmul_op(relu2, W3)

    y4 = y1 + y2
    y = y4 + y3
    y = ht.sigmoid_op(y)

    loss = ht.binarycrossentropy_op(y, y_)
    loss = ht.reduce_mean_op(loss, [0])
    opt = ht.optim.SGDOptimizer(learning_rate=learning_rate)
    train_op = opt.minimize(loss)

    return loss, y, y_, train_op
