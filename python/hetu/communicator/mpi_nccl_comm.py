from ctypes import *
from hetu import ndarray
from hetu.stream import *
import numpy as np
from enum import Enum
import os

def _load_nccl_lib():
    """Load libary in build/lib."""
    curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
    lib_path = os.path.join(curr_path, '../../../build/lib/')
    path_to_so_file = os.path.join(lib_path, "lib_mpi_nccl_runtime_api.so")
    lib = CDLL(path_to_so_file, RTLD_LOCAL)
    return lib

lib_mpi_nccl = _load_nccl_lib()
# lib_mpi_nccl = CDLL("./lib_mpi_nccl_runtime_api.so", RTLD_LOCAL)


class ncclDataType_t(Enum):
    ncclInt8       = 0
    ncclChar       = 0
    ncclUint8      = 1
    ncclInt32      = 2
    ncclInt        = 2
    ncclUint32     = 3
    ncclInt64      = 4
    ncclUint64     = 5
    ncclFloat16    = 6
    ncclHalf       = 6
    ncclFloat32    = 7
    ncclFloat      = 7
    ncclFloat64    = 8
    ncclDouble     = 8
    ncclNumTypes   = 9

class ncclRedOp_t(Enum):
    ncclSum        = 0
    ncclProd       = 1
    ncclMax        = 2
    ncclMin        = 3
    ncclNumOps     = 4

class ncclUniqueId(Structure):
    _fields_=[("internal", (c_int8 * 128))]

class MPI_NCCL_Communicator():

    def __init__(self, stream = None, mpi_init=True, devices=None):
        '''
            mpicomm: the MPI communicator, to use in MPI_Bcast, MPI_Reduce, MPI_Scatter, etc
            ncclcomm: the NCCL communicator, to use in ncclAllReduce ...
            nRanks: the total number of MPI threads
            myRanks: the rank in all MPI threads
            localRank: the rank among the MPI threads in this device
            ncclId: ncclGetUniqueId should be called once when creating a communicator
                    and the Id should be distributed to all ranks in the communicator before calling ncclCommInitRank.
            stream: the stream for NCCL communication
        '''
        self.mpicomm = c_int64(0)
        self.ncclcomm = c_int64(0)
        self.nRanks = c_int32(0)
        self.myRank = c_int32(0)
        self.localRank = c_int32(-1)
        self.ncclId = ncclUniqueId()
        self.device_id = c_int(0)

        if mpi_init:
            self.MPI_Init()
        self.groupComm_flag = False
        self.MPIGetComm()
        self.MPI_Comm_rank()
        self.MPI_Comm_size()
        self.getLocalRank()

        self.device_id.value = devices[self.localRank.value] if devices else self.localRank.value

        if stream == None:
            self.stream = create_stream_handle(ndarray.gpu(self.device_id.value))
        else:
            self.stream = stream

    def MPI_Init(self):
        lib_mpi_nccl.MPIInit()

    def MPI_Finalize(self):
        lib_mpi_nccl.MPIFinalize()

    def MPIGetComm(self):
        lib_mpi_nccl.MPIGetComm(ctypes.byref(self.mpicomm))

    def MPI_Comm_rank(self):
        lib_mpi_nccl.getMPICommRank(ctypes.byref(self.mpicomm), ctypes.byref(self.myRank))

    def MPI_Comm_size(self):
        lib_mpi_nccl.getMPICommSize(ctypes.byref(self.mpicomm), ctypes.byref(self.nRanks))

    def getLocalRank(self):
        lib_mpi_nccl.getLocalRank(ctypes.byref(self.mpicomm), self.nRanks, self.myRank, ctypes.byref(self.localRank))

    def ncclGetUniqueId(self, senderRank = 0):
        lib_mpi_nccl.getNcclUniqueId(ctypes.byref(self.ncclId), self.mpicomm, self.localRank, c_int(senderRank))

    def dlarrayNcclAllReduce(self, input_arr, output_arr, datatype, reduceop, executor_stream = None):
        lib_mpi_nccl.dlarrayAllReduce(input_arr.handle, output_arr.handle, c_int(datatype.value), c_int(reduceop.value), self.ncclcomm, executor_stream.handle if executor_stream else self.stream.handle)

    def dlarrayBroadcast(self, input_arr, output_arr, datatype, root, executor_stream = None):
        if self.groupComm_flag:
            if root not in self.rank_map.keys():
                print("Error: Broadcast root not in Comm group.")
                assert(False)
            root = self.rank_map[root]
        lib_mpi_nccl.dlarrayBroadcast(input_arr.handle, output_arr.handle, c_int(datatype.value), c_int(root), self.ncclcomm, executor_stream.handle if executor_stream else self.stream.handle)

    def dlarrayAllGather(self, input_arr, output_arr, datatype, executor_stream = None):
        lib_mpi_nccl.dlarrayAllGather(input_arr.handle, output_arr.handle, c_int(datatype.value), self.ncclcomm, executor_stream.handle if executor_stream else self.stream.handle)

    def dlarraySend(self, arr, datatype, target, executor_stream = None):
        lib_mpi_nccl.dlarraySend(arr.handle, c_int(datatype.value), c_int(target), self.ncclcomm, executor_stream.handle if executor_stream else self.stream.handle)

    def dlarrayRecv(self, arr, datatype, src, executor_stream = None):
        lib_mpi_nccl.dlarrayRecv(arr.handle, c_int(datatype.value), c_int(src), self.ncclcomm, executor_stream.handle if executor_stream else self.stream.handle)

    def ncclCommInitRank(self):
        '''
            Use partial AllReduce to change here.
            self.nRanks is the number of threads to use ncclallreduce
            self.myRank is the rank among these threads. the value must in [0, self.nRank - 1]
        '''
        lib_mpi_nccl.initNcclCommRank(ctypes.byref(self.ncclcomm), self.nRanks, ctypes.byref(self.ncclId), self.myRank, self.localRank)

    def ncclCommDestroy(self):
        lib_mpi_nccl.commDestroyNccl(ctypes.byref(self.ncclcomm))

    def ncclSetDevice(self, device_id):
        self.device_id.value = device_id
        lib_mpi_nccl.setDevice(self.device_id.value)

    def ncclInit(self):
        self.ncclSetDevice(self.device_id.value)
        self.ncclGetUniqueId()
        self.ncclCommInitRank()

    def ncclFinish(self):
        self.MPI_Finalize()

    def ncclGroupInit(self, group_list):
        self.groupComm_flag = True
        if not isinstance(group_list, list):
            print("Type Error: Group_list should be a list.")
            assert (False)
        if len(set(group_list)) != len(group_list):
            print("Warning: Repeated ranks are found in the group.")
            group_list = list(set(group_list))

        global_rank = self.localRank.value
        size = self.nRanks.value
        group_rank = -1
        group_size = len(group_list)

        self.rank_map=dict()
        if group_size > size:
            print("Error: Too many ranks in the group.")
            assert (False)
        for i in range(group_size):
            if not isinstance(group_list[i],int):
                print("Error: Ranks should be int type.")
                assert (False)
            if group_list[i] > size - 1:
                print("Error: The range of ranks should be [0, size-1].")
                assert(False)
            self.rank_map[group_list[i]] = i

        if global_rank in group_list:
            group_rank = self.rank_map[global_rank]
        self.nRanks = c_int32(group_size)
        self.myRank = c_int32(group_rank)
        self.localRank = c_int32(group_rank)
        self.device_id = c_int(global_rank)
        self.ncclSetDevice(self.device_id.value)
        self.ncclGetUniqueId(senderRank = group_list[0])
        if global_rank not in group_list:
            return
        self.ncclCommInitRank()

def mpi_nccl_communicator(mpi_init=True, devices=None):
    '''

    '''
    return MPI_NCCL_Communicator(mpi_init=mpi_init, devices=devices)

# NCCL_DEBUG=INFO mpirun --allow-run-as-root -np 4 python mpi_nccl_comm.py
if __name__ == "__main__":
    t = mpi_nccl_communicator()
    t.ncclInit()

    arr = np.ones(16)*t.localRank.value
    print("before: = ", arr)
    arr = ndarray.array(arr, ctx = ndarray.gpu(t.device_id.value))
    output_arr = np.zeros(16 * t.nRanks.value)

    output_arr = ndarray.array(output_arr, ctx = ndarray.gpu(t.device_id.value))
    t.dlarrayNcclAllReduce(arr, arr, ncclDataType_t.ncclFloat32, ncclRedOp_t.ncclSum)
    # t.dlarrayBroadcast(arr, ncclDataType_t.ncclFloat32, 0)
    # t.dlarrayAllGather(arr, output_arr, ncclDataType_t.ncclFloat32)

    print("after: = ", arr.asnumpy())

    t.ncclFinish()

