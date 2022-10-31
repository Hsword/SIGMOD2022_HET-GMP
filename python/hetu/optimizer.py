import os
import numpy as np
import ctypes
from . import ndarray
from . import gpu_links as gpu_op
from . import gpu_ops as ht
from .gpu_ops.Node import Op
from .gpu_ops.EmbeddingLookUp import EmbeddingLookUp_Gradient
from .gpu_ops.AllReduceCommunicate import allreduceCommunicate_op
from .gpu_ops.ParameterServerCommunicate import ParameterServerCommunicateOp, parameterServerCommunicate_op
from .gpu_ops.Variable import PlaceholderOp

class Optimizer(object):
    """Optimizers."""
    def __init__(self):
        self.learning_rate = 0
        self.params = None
        self.tensors = None
        self.initiated = False

    @staticmethod
    def get_var_list(loss):
        def topo_sort_dfs(node, visited, var_list):
            if node in visited:
                return
            visited.add(node)
            if isinstance(node, PlaceholderOp) and node.trainable:
                var_list.append(node)
                return
            for n in node.inputs:
                topo_sort_dfs(n, visited, var_list)

        visited = set()
        trainable_vars = []
        if isinstance(loss, list):
            for l in loss:
                topo_sort_dfs(l, visited, trainable_vars)
        else:
            topo_sort_dfs(loss, visited, trainable_vars)
        return trainable_vars

    def initiate_states(self):
        assert not self.initiated, "Optimizer already initiated."
        self.tensors = [node.tensor_value for node in self.params]
        self.initiated = True

    def minimize(self, loss, var_list=None):
        """Return an optimizer op to update parameters.

        Parameters
        ----------
        loss: loss node that we are minimizing.
        var_list: list of nodes that we are taking derivative wrt.

        Returns
        -------
        An optimizer node.

        """
        if not var_list:
            var_list = self.get_var_list(loss)
        self.params = var_list
        grads = ht.gradients(loss, self.params)
        optimizer_node = OptimizerOp(grads, self)
        return optimizer_node


class OptimizerOp(Op):
    def __init__(self, grads, optimizer):
        super().__init__(OptimizerOp, grads, None)
        self.name = "Optimizer_%s" % (optimizer.name)
        self.optimizer = optimizer

    def compute(self, input_vals, output_val, stream_handle=None):
        assert output_val is None
        # For PS op, this input_vals is None
        # PS mode doesn't need local update
        if self.comm_mode != 'PS':
            self.optimizer.update(input_vals, stream_handle)

    def gradient(self, output_grad):
        raise NotImplementedError

    def infer_shape(self, input_shapes):
        return None
    
    def forward_hook(self, config):
        # disable inplace if not lazy execution
        # previously we use array reshape lazy callback to do this, which is deprecated (not efficient)
        for node in self.inputs:
            node.inplace = False

        self.optimizer.initiate_states()
        self.on_cpu = self.on_gpu = None
        self.comm_mode = config.comm_mode
        # some things todo.
        if self.comm_mode != 'PS':
            for i in range(len(self.inputs)):
                # Though the gradients for transfer ops are well defined, 
                # we called gradients in optimizer op before transfer ops are added.
                # So here we also add tranfer ops for gradients update.
                # Could be optimized later.
                if not isinstance(self.inputs[i], ParameterServerCommunicateOp):
                    paramctx = self.optimizer.params[i].ctx
                    self.inputs[i] = super().add_transfer_op(self.inputs[i], paramctx, config.h2d_ops, config.d2h_ops)
                    
    def backward_hook(self, config):
        self.comm_mode = config.comm_mode
        new_inputs = []
        for i, node in enumerate(self.inputs):
            current_strategy = config.node_strategy.get(self.optimizer.params[i], self.comm_mode)
            if current_strategy == 'AllReduce' or (current_strategy == 'Hybrid' and not isinstance(node, EmbeddingLookUp_Gradient)):
                new_inputs.append(ht.allreduceCommunicate_op(node, config.param_allreduce_group.get(self.optimizer.params[i], config.nccl_comm)))
            elif current_strategy == 'PS' or (current_strategy == 'Hybrid' and isinstance(node, EmbeddingLookUp_Gradient)):
                new_inputs.append(ht.parameterServerCommunicate_op(
                    node, self.optimizer.params[i], self.optimizer.get_config()))
            else:
                new_inputs.append(node)
        self.inputs = new_inputs


class SGDOptimizer(Optimizer):
    def __init__(self, learning_rate=0.01):
        super(SGDOptimizer, self).__init__()
        self.learning_rate = learning_rate
        self.name = 'SGD'
    
    def get_config(self):
        return (ctypes.c_int(0), (ctypes.c_float * 1)(self.learning_rate), ctypes.c_int(1))

    def initiate_states(self):
        super().initiate_states()

    def update(self, grads, stream_handle=None):
        assert self.initiated is True
        params_size = len(self.params)
        assert params_size == len(grads)
        for i in range(params_size):
            if grads[i] == None:
                continue
            if self.params[i].on_gpu:
                assert isinstance(self.tensors[i], ndarray.NDArray)
                assert isinstance(grads[i], (ndarray.NDArray, ndarray.IndexedSlices))
                gpu_op.sgd_update(self.tensors[i], grads[i], self.learning_rate, stream_handle)
            else:
                from ._base import DNNL_LIB
                if isinstance(grads[i], ndarray.IndexedSlices):
                    if DNNL_LIB['cpu_SGDOptimizerSparseUpdate']:
                        from .cpu_links import sgd_update_sparse as cpu_sgd_update_sparse
                        cpu_sgd_update_sparse(self.tensors[i], grads[i].indices, grads[i].values, self.learning_rate)
                    else:
                        grads[i].cpu_deduplicate()
                        np_tensor = self.tensors[i].asnumpy()
                        np_tensor[grads[i].indices.asnumpy().astype(np.int)] -= self.learning_rate * grads[i].values.asnumpy()
                        self.tensors[i][:] = np_tensor
                        grads[i].free_deduplicate()
                else:
                    if DNNL_LIB['cpu_SGDOptimizerUpdate']:
                        from .cpu_links import sgd_update as cpu_sgd_update
                        cpu_sgd_update(self.tensors[i], grads[i], self.learning_rate)
                    else:
                        self.tensors[i][:] = self.tensors[i].asnumpy() - self.learning_rate * grads[i].asnumpy()


class MomentumOptimizer(Optimizer):
    def __init__(self, learning_rate=0.01, momentum=0.9, nesterov=False):
        super(MomentumOptimizer, self).__init__()
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.nesterov = nesterov
        self.name = "Momentum"

    def get_config(self):
        return (ctypes.c_int(self.nesterov + 1), (ctypes.c_float * 2)(self.learning_rate, self.momentum), ctypes.c_int(2))

    def initiate_states(self):
        super().initiate_states()
        self.velocity = []
        for t in self.tensors:
            self.velocity.append(None if t is None else ndarray.array(np.zeros(t.shape, dtype=np.float32), t.ctx))

    def update(self, grads, stream_handle=None):
        assert self.initiated is True
        params_size = len(self.params)
        assert params_size == len(grads)
        for i in range(params_size):
            if grads[i] == None:
                continue
            if self.params[i].on_gpu:
                assert isinstance(self.tensors[i], ndarray.NDArray)
                assert isinstance(grads[i], (ndarray.NDArray, ndarray.IndexedSlices))
                assert isinstance(self.velocity[i], ndarray.NDArray)
                gpu_op.momentum_update(self.tensors[i], grads[i], self.velocity[i], self.learning_rate, self.momentum,
                                       self.nesterov, stream_handle)
            else:
                if isinstance(grads[i], ndarray.IndexedSlices):
                    raise NotImplementedError
                else:
                    from ._base import DNNL_LIB
                    if DNNL_LIB['cpu_MomentumOptimizerUpdate']:
                        from .cpu_links import momentum_update as cpu_momentum_update
                        cpu_momentum_update(self.tensors[i], grads[i], self.velocity[i],self.learning_rate,self.momentum,
                                        self.nesterov)
                    else:
                        if self.nesterov:
                            lr_grads = -self.learning_rate * grads[i].asnumpy()
                            self.velocity[i][:] = self.momentum * (self.velocity[i].asnumpy() + lr_grads)
                            self.tensors[i][:] = self.tensors[i].asnumpy() + self.velocity[i].asnumpy() + lr_grads
                        else:
                            self.velocity[i][:] = self.momentum * self.velocity[i].asnumpy() - self.learning_rate * grads[i].asnumpy()
                            self.tensors[i][:] = self.tensors[i].asnumpy() + self.velocity[i].asnumpy()


class AdaGradOptimizer(Optimizer):
    def __init__(self, learning_rate=0.01, initial_accumulator_value=0.0, eps=1e-7):
        assert learning_rate >= 0, \
            "learning rate must be non-negative"
        assert initial_accumulator_value >= 0.0, \
            "initial accumulator value must be non-negative"
        assert eps > 0.0, \
            "epsilon must be positive"
        super(AdaGradOptimizer, self).__init__()
        self.learning_rate = learning_rate
        self.initial_accumulator_value = initial_accumulator_value
        self.eps = eps
        self.name = "AdaGrad"
    
    def get_config(self):
        return (ctypes.c_int(3), (ctypes.c_float * 3)(self.learning_rate, self.initial_accumulator_value, self.eps), ctypes.c_int(3))

    def initiate_states(self):
        super().initiate_states()
        self.accumulator_value = []
        for t in self.tensors:
            self.accumulator_value.append(None if t is None else ndarray.array(np.full(t.shape, self.initial_accumulator_value), t.ctx))

    def update(self, grads, stream_handle=None):
        assert self.initiated is True
        params_size = len(self.params)
        assert params_size == len(grads)
        for i in range(params_size):
            if grads[i] == None:
                continue
            if self.params[i].on_gpu:
                assert isinstance(self.tensors[i], ndarray.NDArray)
                assert isinstance(grads[i], (ndarray.NDArray, ndarray.IndexedSlices))
                gpu_op.adagrad_update(self.tensors[i], grads[i], self.accumulator_value[i], self.learning_rate, self.eps,
                                      stream_handle)
            else:
                if isinstance(grads[i], ndarray.IndexedSlices):
                    raise NotImplementedError
                else:
                    from ._base import DNNL_LIB
                    if DNNL_LIB['cpu_AdaGradOptimizerUpdate']:
                        from .cpu_links import adagrad_update as cpu_adagrad_update
                        cpu_adagrad_update(self.tensors[i], grads[i], self.accumulator_value[i],self.learning_rate,self.eps)
                    else:
                        self.accumulator_value[i][:] = self.accumulator_value[i].asnumpy() + np.power(grads[i].asnumpy(), 2)
                        self.tensors[i][:] = \
                            self.tensors[i].asnumpy() - self.learning_rate * grads[i].asnumpy() / (np.sqrt(self.accumulator_value[i].asnumpy()) + self.eps)


class AdamOptimizer(Optimizer):
    def __init__(self, learning_rate=0.01, beta1=0.9, beta2=0.999, epsilon=1e-7):
        super(AdamOptimizer, self).__init__()
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta1_t = 1.0
        self.beta2 = beta2
        self.beta2_t = 1.0
        self.epsilon = epsilon
        self.name = "Adam"
    
    def get_config(self):
        return (ctypes.c_int(4), (ctypes.c_float * 4)(self.learning_rate, self.beta1, self.beta2, self.epsilon), ctypes.c_int(4))

    def initiate_states(self):
        super().initiate_states()
        self.m = []
        self.v = []
        for t in self.tensors:
            self.m.append(None if t is None else ndarray.array(np.zeros(t.shape), t.ctx))
            self.v.append(None if t is None else ndarray.array(np.zeros(t.shape), t.ctx))

    def update(self, grads, stream_handle=None):
        assert self.initiated is True
        params_size = len(self.tensors)
        assert params_size == len(grads)
        self.beta1_t *= self.beta1
        self.beta2_t *= self.beta2
        for i in range(params_size):
            if grads[i] == None:
                continue
            if self.params[i].on_gpu:
                assert isinstance(self.tensors[i], ndarray.NDArray)
                assert isinstance(grads[i], (ndarray.NDArray, ndarray.IndexedSlices))
                assert isinstance(self.m[i], ndarray.NDArray)
                assert isinstance(self.v[i], ndarray.NDArray)
                gpu_op.adam_update(self.tensors[i], grads[i], self.m[i], self.v[i], self.learning_rate, self.beta1,
                                   self.beta2, self.beta1_t, self.beta2_t, self.epsilon, stream_handle)
            else:
                if isinstance(grads[i], ndarray.IndexedSlices):
                    raise NotImplementedError
                else:
                    from ._base import DNNL_LIB
                    if DNNL_LIB['cpu_AdamOptimizerUpdate']:
                        from .cpu_links import adam_update as cpu_adam_update
                        cpu_adam_update(self.tensors[i], grads[i], self.m[i],self.v[i], self.learning_rate, self.beta1,
                                    self.beta2, self.beta1_t, self.beta2_t, self.epsilon)
                    else:
                        self.m[i][:] = self.beta1 * self.m[i].asnumpy() + (1 - self.beta1) * grads[i].asnumpy()
                        self.v[i][:] = self.beta2 * self.v[i].asnumpy() + (1 - self.beta2) * grads[i].asnumpy() * grads[i].asnumpy()
                        mc = self.m[i].asnumpy() / (1 - self.beta1_t)
                        vc = self.v[i].asnumpy() / (1 - self.beta2_t)
                        self.tensors[i][:] = self.tensors[i].asnumpy() - self.learning_rate * mc / (np.sqrt(vc) + self.epsilon)
