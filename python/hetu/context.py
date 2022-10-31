from .ndarray import cpu, gpu, DLContext, is_gpu_ctx
import contextlib
import socket
import re

class DeviceGroup(object):
    def __init__(self, ctxs):
        self._contexts = self.parse_contexts(ctxs)
    
    @classmethod
    def parse_contexts(cls, ctxs):
        if isinstance(ctxs, DeviceGroup):
            return ctxs
        if isinstance(ctxs, str):
            ctxs = re.split(';|,| +',ctxs.lower())
        if not isinstance(ctxs, list):
            ctxs = [ctxs]
        new_ctxs = []
        local_hostname = socket.gethostname()
        for c in ctxs:
            if isinstance(c, str):
                c = c.lower().split(':')
                if len(c) == 3 and c[0] != local_hostname:
                    continue
                assert c[-2] in ('cpu', 'gpu'), 'Context invalid: %s' % c
                idx = int(c[-1])
                c = cpu(idx) if c[-2] == 'cpu' else gpu(idx)
            assert isinstance(c, DLContext), 'Context invalid: %s' % c
            new_ctxs.append(c)
        return new_ctxs
    
    def index(self, ctx):
        return self._contexts.index(ctx)

    def __getitem__(self, key):
        return self._contexts[key]
    
    def __iter__(self):
        return iter(self._contexts)
    
    def __len__(self):
        return len(self._contexts)
    
    def __repr__(self):
        result = 'DeviceGroup('
        for c in self._contexts:
            result += '%s, ' % c
        result += ')'
        return result

    def __hash__(self):
        if not hasattr(self, 'hash'):
            self.hash = hash(tuple(sorted(self._contexts, key=lambda x: x.device_id)))
        return self.hash

    def __eq__(self, other):
        return hash(self) == hash(other)


class ContextStack(object):
    def __init__(self):
        self._stack = []

    def peek(self):
        return self._stack[-1] if self._stack else None

    def push(self, ctx):
        return self._stack.append(ctx)

    def pop(self):
        self._stack.pop()

_default_ctx_stack = ContextStack()

def get_current_context():
    return _default_ctx_stack.peek()

@contextlib.contextmanager
def context(ctx):
    try:
        ctx = DeviceGroup(ctx)
        _default_ctx_stack.push(ctx)
        yield ctx
    finally:
        _default_ctx_stack.pop()


def get_launch_config_by_traverse_nodes(node_list, default_ctx):
    node_strategy = dict()
    devices = set(iter(default_ctx))
    launchPS = len(default_ctx) > 1 and any([not is_gpu_ctx(c) for c in default_ctx])
    launchMPI = (not launchPS) and len(default_ctx) > 1
    nrank = len([c for c in default_ctx if is_gpu_ctx(c)])
    for node in node_list:
        traverse_dfs(node, node_strategy, devices, nrank)
    launchPS = launchPS or any([x == 'PS' for x in node_strategy.values()])
    launchMPI = launchMPI or any([x == 'AllReduce' for x in node_strategy.values()])
    return launchMPI, launchPS, node_strategy, devices

def traverse_dfs(node, node_strategy, devices, nrank):
    if node in node_strategy:
        return
    strategy = None
    if node.raw_ctx is not None and len(node.raw_ctx) > 1 and any([not is_gpu_ctx(c) for c in node.raw_ctx]):
        strategy = 'PS'
    elif node.raw_ctx is not None and len(node.raw_ctx) > 1:
        strategy = 'AllReduce'
    node_strategy[node] = strategy
    devices.update(iter(node.raw_ctx))
    local_nrank = nrank if node.raw_ctx is None else len([c for c in node.raw_ctx if is_gpu_ctx(c)])
    assert local_nrank in (0, nrank), 'Number of workers not consist: (%d, %d).' % (local_nrank, nrank)
    for n in node.inputs:
        traverse_dfs(n, node_strategy, devices, nrank)


def assign_context_by_traverse_nodes(node_list, ctx, mpi_comm, p2p_stream):
    def assign_ctx(node):
        from .dataloader import DataloaderOp
        from .optimizer import OptimizerOp
        from .gpu_ops.PipelineSend import pipeline_send_op
        from .gpu_ops.PipelineReceive import pipeline_receive_op
        from .gpu_ops.Variable import PlaceholderOp
        if node in visited:
            return
        visited.add(node)
        if isinstance(node, DataloaderOp):
            return
        elif isinstance(node, OptimizerOp):
            nonlocal opt
            assert opt is None, 'Multiple optimizer is invalid.'
            opt = node
            for n in node.inputs:
                assign_ctx(n)
            my_eval_nodes.append(node)
        else:
            is_my_node = ctx in node.raw_ctx
            for i, n in enumerate(node.inputs):
                if isinstance(n, DataloaderOp):
                    if is_my_node and n in node_list and n not in my_eval_nodes:
                        my_eval_nodes.append(n)
                    continue
                assign_ctx(n)

                # we assume that in model parallel + data parallel mode,
                # devices number of each stage is equal
                # the device in correspondent place will communicate with each other
                # TODO: not support following case: context(1,5) -> context(5,1); context(1,5) -> context(3,1)
                # solution: modify following is_my_node logic to support
                # TODO: not support the case that each process has different group init numbers, since there is an AllGather in mpi_nccl_comm's init
                # solution: modify mpi_nccl_comm class, so that the MPI part only process once while nccl has several groups
                assert len(node.raw_ctx) == len(n.raw_ctx), \
                    'In pipeline + data parallel, devices number of each stage should be equal!'

                if is_my_node and ctx not in n.raw_ctx:
                    my_pos = node.raw_ctx.index(ctx)
                    target_id = n.raw_ctx[my_pos].device_id
                    key = (n, target_id)
                    if key not in recv_src:
                        recv_src[key] = pipeline_receive_op(target_id, mpi_comm, stream=p2p_stream, ctx=ctx)
                    node.inputs[i] = recv_src[key]
                elif not is_my_node and ctx in n.raw_ctx:
                    my_pos = n.raw_ctx.index(ctx)
                    target_id = node.raw_ctx[my_pos].device_id
                    key = (n, target_id)
                    if key not in send_dst:
                        send_dst[key] = pipeline_send_op(n, target_id, mpi_comm, stream=p2p_stream, ctx=ctx)
                        my_eval_nodes.append(send_dst[key])
            if is_my_node:
                node.ctx = ctx
                if node in node_list:
                    my_eval_nodes.append(node)
                if isinstance(node, PlaceholderOp) and node.trainable:
                    trainable_params.append(node)

    opt = None
    trainable_params = []
    send_dst = {}
    recv_src = {}
    visited = set()
    my_eval_nodes = []
    for node in node_list:
        assign_ctx(node)

    has_send_recv = send_dst != {} or recv_src != {}
    if opt and trainable_params:
        original_params = opt.optimizer.params
        indices = [original_params.index(param) for param in trainable_params]
        opt.optimizer.params = trainable_params
        grads = [opt.inputs[index] for index in indices]
        opt.inputs = grads
        opt.ctx = ctx
    elif opt:
        my_eval_nodes.remove(opt)
    return my_eval_nodes, trainable_params, has_send_recv

