#!/usr/bin/env python
# coding=UTF-8
'''
@Author: Jianzhou Zhao
LastEditors: your name
@Description: This is a wrapped code for the mpi4py
@Date: 2019-04-19 22:00:22
LastEditTime: 2024-01-19 15:14:00
'''

import contextlib
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

class CollectiveError(RuntimeError):
    """Raised on every rank when local work failed on at least one rank (several ranks only)."""


class NonFiniteResult(FloatingPointError):
    """Non-finite values found by a check on data a collective call made identical on
    every rank, so every rank raises it alike.  Only for such checks: a plain
    FloatingPointError says nothing about the other ranks."""


@contextlib.contextmanager
def _collective(what, comm):
    """Implementation of collective() for a given communicator (the mpiutils wrappers
    pass their own); rank, size and the message exchange all come from comm."""
    if comm is None or comm.size == 1:
        yield
        return
    err = None
    try:
        yield
    except Exception as exc:              # KeyboardInterrupt/SystemExit are not intercepted
        err = exc
    msgs = [m for m in comm.allgather(None if err is None else
                                      f"rank {comm.rank}: {type(err).__name__}: {err}") if m]
    if msgs:
        raise CollectiveError(f"{what} failed on {len(msgs)} rank(s): " + "; ".join(msgs[:5])) from err


def collective(what):
    """Guard for the local work between two collective calls on COMM_WORLD.

    Every rank must enter, also one with no local work, and the block must not
    contain a collective call.  Single rank: the block runs unguarded and an
    exception propagates unchanged.  Several ranks: an exception in the block
    is held, all ranks exchange one message (allgather), and if any rank failed
    every rank raises CollectiveError naming the failed ranks and their
    exception types; the failing rank chains its own exception.  Not covered:
    a process killed by a signal, a native crash, a failure inside an MPI call."""
    return _collective(what, comm)


def allgather(obj, comm=comm):
    if comm is not None and comm.size > 1:
        return comm.allgather(obj)
    else:
        return [obj]

def barrier(comm=comm):
    """Real synchronisation.  bcast is not one: the root returns as soon as
    the (small) message is sent, before the other ranks have arrived."""
    if comm is not None and comm.size > 1:
        comm.Barrier()

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
        with _collective("Allreduce (receive buffer)", comm):
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

def _as_double(name, arr):
    """Buffers sent as MPI.DOUBLE must really be contiguous float64."""
    from numpy import float64
    if arr.dtype != float64 or not arr.flags.c_contiguous:
        raise TypeError(f"{name}: MPI.DOUBLE buffer must be C-contiguous float64, got "
                        f"{arr.dtype}{'' if arr.flags.c_contiguous else ' (non-contiguous)'}")
    return arr

def Allgatherv(sendobj, recvobj, count, disp, comm=comm):
    if comm is not None:
        with _collective("Allgatherv (buffer check)", comm):
            send = _as_double("Allgatherv send", sendobj)
            recv = _as_double("Allgatherv recv", recvobj)
        comm.Allgatherv(send, [recv, count, disp, MPI.DOUBLE])
    else:
        recvobj[...] = sendobj.reshape(recvobj.shape)

# def Scatterv(sendobj, recvobj, root, comm=comm):
#     if comm is not None and comm.size > 1:
#         comm.Scatterv(sendobj, recvobj, root=root)
#     else:
#         recvobj = sendobj

def devide_array(sendobj, root):
    """
    Devide sendobj into size part along the first axis.  The data are sent as
    MPI.DOUBLE, so the input is converted to C-contiguous float64 first (a
    float32 or integer array would otherwise be read with the wrong layout).
    """
    from numpy import zeros, ascontiguousarray, float64, prod

    with _collective("devide_array (preparation)", comm):
        sendobj = ascontiguousarray(sendobj, dtype=float64)
        if sendobj.ndim < 1:
            raise ValueError("devide_array: need an array with at least one axis")
        if comm is not None:
            total_count = len(sendobj)
            " For array more than 1 dimension, we only devide the first axis "
            coeff = int(prod(sendobj.shape[1:], dtype=int))
            count = [coeff*(total_count//size+1) if irank <= total_count % size-1
                     else coeff*(total_count//size) for irank in range(size)]
            disp = [sum(count[:irank]) for irank in range(size)]
            recvobj = zeros(count[rank])
    if comm is None:
        return sendobj, [len(sendobj)], [0]
    comm.Scatterv([sendobj, count, disp, MPI.DOUBLE], recvobj, root=root)
    return recvobj.reshape(-1, *sendobj.shape[1:]), [ic//coeff for ic in count], [id//coeff for id in disp]
