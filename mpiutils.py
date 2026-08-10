#!/usr/bin/env python
# coding=UTF-8
'''
@Author: Jianzhou Zhao
LastEditors: your name
@Description: This is a wrapped code for the mpi4py
@Date: 2019-04-19 22:00:22
LastEditTime: 2024-01-19 15:14:00
'''

import sys
import warnings

rank = 0
size = 1
comm = None
world = None
rank0 = True

# Try to setup MPI and get the comm, rank and size.
# If not they should end up as rank=0, size=1.
try:

    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    double = MPI.DOUBLE

    if comm is not None and size > 1:
        print("Starting MPI rank=%i [size=%i]" % (rank, size))
        sys.stdout.flush()

    rank0 = True if rank == 0 else False

except ImportError:

    warnings.warn("Warning: mpi4py not installed.")

def bcast(obj, root=0, comm=comm):
    if comm is not None and comm.size > 1:
        return comm.bcast(obj, root=root)
    else:
        return obj

def reduce(obj, root=0, op=None, comm=comm):
    if comm is not None and comm.size > 1:
        return comm.reduce(obj, root=root, op=(op or MPI.SUM))
    else:
        return obj

def allreduce(obj, op=None, comm=comm):
    if comm is not None and comm.size > 1:
        return comm.allreduce(obj, op=(op or MPI.SUM))
    else:
        return obj

def gather(obj, root, comm=comm):
    if comm is not None and comm.size > 1:
        return comm.gather(obj, root=root)
    else:
        return obj

# 以下大写版包装目前没有任何代码调用，且串行回退分支只是重绑定局部变量名
# （recvobj = sendobj 不会写回调用方缓冲区，串行时拿到全零）。
# 先注释掉；需要启用时把 else 分支改成 recvobj[...] = sendobj.reshape(recvobj.shape)。
#
# def Reduce(sendobj, recvobj, root=0, op=None, comm=comm):
#     if comm is not None and comm.size > 1:
#         comm.Reduce(sendobj, recvobj, root=root, op=(op or MPI.SUM))
#     else:
#         recvobj = sendobj
#
def Allreduce(sendobj, op=None, comm=comm):
    from numpy import zeros_like
    if comm is not None and comm.size > 1:
        recvobj = zeros_like(sendobj)
        comm.Allreduce(sendobj, recvobj, op=(op or MPI.SUM))
        return recvobj
    else:
        return sendobj
#
# def Gather(sendobj, recvobj, root, comm=comm):
#     if comm is not None and comm.size > 1:
#         comm.Gather(sendobj, recvobj, root=root)
#     else:
#         recvobj = sendobj

def Allgatherv(sendobj, recvobj, count, disp, comm=comm):
    if comm is not None:
        comm.Allgatherv(sendobj, [recvobj,count,disp,MPI.DOUBLE])
    else:
        recvobj[...] = sendobj.reshape(recvobj.shape)

# def Scatterv(sendobj, recvobj, root, comm=comm):
#     if comm is not None and comm.size > 1:
#         comm.Scatterv(sendobj, recvobj, root=root)
#     else:
#         recvobj = sendobj

def devide_array(sendobj, root):
    """
    Devide sendobj into size part.
    """
    from numpy import zeros
    from functools import reduce

    if comm is None:
        return sendobj, [len(sendobj)], [0]

    total_count = len(sendobj)
    " For array more than 1 dimension, we only devide the first axis "
    coeff = reduce(lambda x, y: x*y, sendobj.shape[1:])
    count = [coeff*(total_count//size+1) if irank <= total_count % size-1
             else coeff*(total_count//size) for irank in range(size)]
    disp = [sum(count[:irank]) for irank in range(size)]
    recvobj = zeros(count[rank])
    comm.Scatterv([sendobj, count, disp, MPI.DOUBLE], recvobj, root=root)
    return recvobj.reshape(-1, *sendobj.shape[1:]), [ic//coeff for ic in count], [id//coeff for id in disp]

