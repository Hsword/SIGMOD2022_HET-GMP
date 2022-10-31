from __future__ import absolute_import
from .Node import Op

import numpy as np
import time
import os

class SharedTableOp(Op):
    def __init__(self, node, rank, nrank, table_shape, learning_rate, ctx,
        root_arr=None, storage_arr=None, bounds=(0, 0), init = None):

        from .. import ndarray, placeholder_op

        self.fake_param = placeholder_op("fake", value=ndarray.empty([1], ctx=ctx))

        super().__init__(SharedTableOp, [self.fake_param], ctx)

        import hetuCTR

        assert(ndarray.is_gpu_ctx(ctx))
        if "HETUCTR_INIT_ADDR" in os.environ:
            ip = os.environ["HETUCTR_INIT_ADDR"]
        else:
            ip = "127.0.0.1"
        port = 23456

        self.tbl_length, self.tbl_width = table_shape
        pull_bound, push_bound = bounds

        if not init:
            init = hetuCTR.Initializer(hetuCTR.InitType.Normal, 0 , 0.01)
        if root_arr is None:
            random_state = np.random.get_state()
            np.random.seed(0)
            root_arr = np.random.randint(0, nrank, self.tbl_length)
            np.random.set_state(random_state)
        if storage_arr is None:
            storage_arr = np.where(root_arr == rank)[0]

        # initialize dataloaders
        self.dl_node = node
        for d in node.dataloaders.values():
            d.init_states()

        self.table = hetuCTR.HetuTable(
            rank=rank, nrank=nrank, device_id=ctx.device_id, ip=ip, port=port,
            pull_bound = pull_bound, push_bound = push_bound, init=init, learning_rate=learning_rate,
            length = self.tbl_length, width = self.tbl_width,
            root_arr = root_arr, storage_arr = storage_arr, verbose=0
        )
        self.pipeline_ready = False

        self.task = None

    def compute(self, input_vals, output_val, stream_handle=None, inference=False):
        if self.task:
            self.task.wait()

        self.batch_idx = self.dl_node.get_arr("validate") if inference else self.dl_node.get_arr("train")
        self.next_batch_idx  = self.dl_node.get_next_arr("validate") if inference else self.dl_node.get_next_arr("train")

        if not self.pipeline_ready or inference:
            self.lookup_output = output_val
            self.table.preprocess(self.batch_idx.ctypes.data, np.prod(self.batch_idx.shape))
            self.table.push_pull(0, output_val.handle.contents.data)

        if inference:
            self.pipeline_ready = False
        else:
            self.task = self.table.async_preprocess(self.next_batch_idx.ctypes.data, np.prod(self.next_batch_idx.shape))
            assert(output_val is self.lookup_output)

    def compute_backward(self, gradients):
        self.pipeline_ready = True
        self.task = self.table.async_push_pull(gradients.handle.contents.data, self.lookup_output.handle.contents.data)

    def gradient(self, output_grad):
        self.grad_node = SharedTableGradientOp(output_grad, self, ctx=self.ctx)
        return [self.grad_node]

    def infer_shape(self, input_shapes):
        output_shape = list(self.dl_node.get_cur_shape("train")) + [self.tbl_width]
        return tuple(output_shape)


class SharedTableGradientOp(Op):
    def __init__(self, output_grad, forward_node, ctx=None):
        super().__init__(SharedTableGradientOp, [output_grad], ctx)
        self.forward_node = forward_node

    def compute(self, input_vals, output_val, stream_handle=None):
        # we have to make sure gradient is ready
        if stream_handle:
            stream_handle.sync()
        self.forward_node.compute_backward(input_vals[0])

    def gradient(self, output_grad):
        raise NotImplementedError

    def infer_shape(self, input_shapes):
        return (1, )
