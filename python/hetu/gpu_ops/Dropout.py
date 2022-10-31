from __future__ import absolute_import
from .Node import Op
import ctypes
import numpy as np
from .._base import DNNL_LIB
from ..cpu_links import dropout as cpu_dropout
from ..cpu_links import dropout_gradient as cpu_dropout_gradient
from ..gpu_links import dropout_gradient
from ..gpu_links import dropout


class DropoutOp(Op):
    def __init__(self, node_in, keep_prob, ctx=None):
        super().__init__(DropoutOp, [node_in], ctx)
        self.seed = ctypes.c_ulonglong(0)
        # self.reserve_size = ctypes.c_int(0)
        # self.reserve_space = ctypes.c_void_p(0)
        self.mask = None
        self.keep_prob = keep_prob
        # self.flag = 1
        # self.input_shape = None

    def compute(self, input_vals, output_val, stream_handle=None, inference=False):
        if inference == False:
            if self.on_cpu:
                if DNNL_LIB['cpu_Dropout']:
                    cpu_dropout(input_vals[0], self.keep_prob, output_val)
                else:
                    np.random.seed(self.seed.value)
                    if self.mask is None:
                        #self.mask = np.random.binomial(1, self.keep_prob, size=input_vals[0].shape)
                        self.mask=np.random.uniform(0,1.0,input_vals[0].shape)>=(1-self.keep_prob)
                    output_val[:] = dropout_np(input_vals[0].asnumpy(), self.keep_prob, output_val, self.mask)
            else:
                dropout(input_vals[0], 1 - self.keep_prob, output_val, self.seed, stream_handle)
                # CuDNN_Dropout(input_vals[0], self.keep_prob, output_val, self.reserve_size, self.reserve_space, self.flag, stream_handle)
                # self.flag = 0

    def gradient(self, output_grad):
        return [dropout_gradient_op(output_grad, self.keep_prob, self, ctx=self.raw_ctx)]

    def infer_shape(self, input_shapes):
        # if self.input_shape != tuple(input_shapes[0]):
        #     self.input_shape = tuple(input_shapes[0])
        #     # self.flag = 2
        return input_shapes[0]


class Dropout_GradientOp(Op):
    def __init__(self, node_in, keep_prob, forward_node, ctx=None):
        super().__init__(Dropout_GradientOp, [node_in], ctx)
        self.forward_node = forward_node
        self.keep_prob = keep_prob

    def compute(self, input_vals, output_val, stream_handle=None):
        if self.on_cpu:
            if DNNL_LIB['cpu_Dropout_Gradient']:
                cpu_dropout_gradient(input_vals[0], self.keep_prob, output_val)
            else:
                output_val[:] = dropout_np_gradient(input_vals[0].asnumpy(), self.keep_prob, self.forward_node.mask)
        else:
            dropout_gradient(input_vals[0], 1 - self.keep_prob, output_val, self.forward_node.seed, stream_handle)
            # CuDNN_Dropout_gradient(input_vals[0], self.keep_prob, output_val, self.forward_node.reserve_size,
            #                        self.forward_node.reserve_space, stream_handle)

    def gradient(self, output_grad):
        raise NotImplementedError

    def infer_shape(self, input_shapes):
        return input_shapes[0]


def dropout_op(node_in, keep_prob, ctx=None):
    """Drops elements of input variable randomly.
    Parameters:
    ----
    node_in : Node
        Input variable.
    keep_prob : float
        Probability of the results to be kept.
    Returns:
    ----
    A new Node instance created by Op.
    """
    return DropoutOp(node_in, keep_prob, ctx=ctx)


def dropout_gradient_op(node_in, keep_prob, forward_node, ctx=None):
    """Gradient node of dropout operation.
    Parameters:
    ----
    node_in : Node
        Input variable.
    keep_prob : float
        Probability of the results to be kept.
    Returns:
    ----
    A new Node instance created by Op.
    """
    return Dropout_GradientOp(node_in, keep_prob, forward_node, ctx=ctx)


def dropout_np(inputs, keep_prob, out_arr, mask):
    # outputs = inputs
    # outputs *= mask
    # outputs = outputs * (1 / keep_prob)
    # return outputs
    return mask*inputs*(1/keep_prob)

def dropout_np_gradient(in_gradient_y, keep_prob,mask):
    out_grads = in_gradient_y
    out_grads *= mask * (1 / keep_prob)
    return out_grads
