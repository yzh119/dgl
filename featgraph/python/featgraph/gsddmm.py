""" The compute function and schedules for GSDDMM kernels written in TVM. """
import tvm
from tvm import te
from tvm import topi
from tvm.topi.utils import prod, ravel_index, unravel_index
from tvm.tir import IntImm, Max
from utils import binary_op_map

__all__ = ['gsddmm']


class TargetCode:
    SRC = 0
    DST = 1
    EDGE = 2


def _sddmm_compute(out_shp, binary_op,
                   lhs, rhs, lhs_idx, rhs_idx):
    reduce_size = lhs.shape[-1] if binary_op == 'dot' else 1
    feat_len = prod(out_shp[1:])
    feat_len *= reduce_size
    if binary_op == 'dot':
        k = te.reduce_axis((0, reduce_size), name='k')
        def dot_edge_func(*args):
            eid = args[0]
            fid = ravel_index(args[1:], out_shp[1:])
            fid *= reduce_size
            lval = lhs.__getitem__((lhs_idx(eid),) + args[1: -1] + (k,))
            rval = rhs.__getitem__((rhs_idx(eid),) + args[1: -1] + (k,))
            return te.sum(lval * rval, axis=k)
        out = te.compute(out_shp, dot_edge_func, name='out')
    else:
        def edge_func(*args):
            eid = args[0]
            fid = ravel_index(args[1:], out_shp[1:])
            lval = lhs.__getitem__((lhs_idx(eid),) + args[1:])
            rval = rhs.__getitem__((rhs_idx(eid),) + args[1:])
            return binary_op_map[binary_op](lval, rval)
        out = te.compute(out_shp, edge_func, name='out')
    return out


def _sddmm_cuda_general(sched, out):
    out_len = prod(out.shape[1:])
    edge_axis = out.op.axis[0]
    feat_axis = sched[out].fuse(*out.op.axis[1:])
    #ntx = tvm.autotvm.task.space.get_pow2s(out_len)[-1]
    #ntx = 1024 if ntx > 1024 else ntx
    #nty = 1024 // ntx
    ntx = 32
    nty = 32
    feat_outer, feat_inner = sched[out].split(feat_axis, factor=ntx)
    edge_outer, edge_inner = sched[out].split(edge_axis, factor=nty)
    sched[out].bind(feat_inner, te.thread_axis('threadIdx.x'))
    sched[out].bind(feat_outer, te.thread_axis('blockIdx.y'))
    sched[out].bind(edge_inner, te.thread_axis('threadIdx.y'))
    sched[out].bind(edge_outer, te.thread_axis('blockIdx.x'))


def _sddmm_cuda_tree_reduce(sched, out):
    edge_axis = out.op.axis[0]
    reduce_axis = out.op.reduce_axis[0]
    # sched[out].bind(reduce_axis, te.thread_axis('threadIdx.x'))
    # sched[out].bind(edge_axis, te.thread_axis('blockIdx.x'))
    _, red_inner = sched[out].split(reduce_axis, factor=32)
    edge_outer, edge_inner = sched[out].split(edge_axis, factor=32)
    sched[out].bind(red_inner, te.thread_axis('threadIdx.x'))
    sched[out].bind(edge_inner, te.thread_axis('threadIdx.y'))
    sched[out].bind(edge_outer, te.thread_axis('blockIdx.x'))


def gsddmm(binary_op,
           lhs_code, rhs_code,
           indice_type, feat_type,
           lhs_target=TargetCode.SRC, rhs_target=TargetCode.DST,
           schedule_type="tree",
           target='cuda'):
    """
    Compile SDDMM kernel using TVM. 

    Parameters
    ----------
    binary_op : str
        Type of binary operatiin, could be ``add``, ``sub``, ``mul``,
        ``div`` or ``dot``.
    lhs_code : str
        A string with length d (the rank of lhs operand) composed of ``1`` and ``x``
        that indicates whether each dimension needs broadcasting or not.
    rhs_code : str
        A string with length d (the rank of rhs operand) composed of ``1`` and ``x``
        that indicates whether each dimension needs broadcasting or not.
    indice_type : str
        Type of graph indices, could be ``int32`` or ``int64``.
    feat_type : str
        Type of features, could be ``float16``/``float32``/``float64``
        or ``int32``/``int64``.
    lhs_target : TargetCode
        Indicates the left-hand-side tensor's target.
    rhs_target : TargetCode
        Indicates the right-hand-side tensor's target.
    schedule_type : str
        Specifies the schedule type, could be either ``tree`` or ``general``.
    target : str
        Indicates where kernels are run, i.e. CPU or GPU.

    Returns
    -------
    IRModule, representing compiled kernel. 
    """
    num_rows = te.var('num_rows', indice_type)
    num_cols = te.var('num_cols', indice_type)
    nnz = te.var('nnz', indice_type)

    # placeholder for sparse matrix
    adj_row_indices = te.placeholder((nnz,), indice_type, 'adj_row_indices')
    adj_col_indices = te.placeholder((nnz,), indice_type, 'adj_col_indices')

    # placeholder for dense features
    def create_placeholder(target, feat_shp, name):
        if target == TargetCode.SRC:
            return te.placeholder((num_rows,) + feat_shp, feat_type, name)
        elif target == TargetCode.EDGE:
            return te.placeholder((nnz,) + feat_shp, feat_type, name)
        elif target == TargetCode.DST:
            return te.placeholder((num_cols,) + feat_shp, feat_type, name)
        else:
            raise DGLError('Unknown target')

    assert len(lhs_code) == len(rhs_code), "lhs code must have equal length with rhs code"
    ndim = len(lhs_code)
    out_feat_shp = [te.var('d{}'.format(i), indice_type) for i in range(ndim)]
    lhs_feat_shp = [di if ci == 'x' else IntImm(indice_type, 1) for ci, di in zip(lhs_code, out_feat_shp)]
    rhs_feat_shp = [di if ci == 'x' else IntImm(indice_type, 1) for ci, di in zip(rhs_code, out_feat_shp)]
    lhs = create_placeholder(lhs_target, tuple(lhs_feat_shp), 'lhs')
    rhs = create_placeholder(rhs_target, tuple(rhs_feat_shp), 'rhs')

    # idx wrapper for corresponding target
    idx_target = {
        TargetCode.SRC: lambda eid: adj_row_indices[eid],
        TargetCode.EDGE: lambda eid: eid,
        TargetCode.DST: lambda eid: adj_col_indices[eid]
    }

    # compute
    out = _sddmm_compute([nnz] + out_feat_shp,
                         binary_op, lhs, rhs,
                         idx_target[lhs_target], idx_target[rhs_target])

    # schedule
    sched = te.create_schedule(out.op)

    if target == 'cuda':
        # cuda schedule
        if schedule_type == 'tree':
            assert binary_op == 'dot', "Tree reduction is only applicable to dot product."
            _sddmm_cuda_tree_reduce(sched, out)
        elif schedule_type == 'general':
            _sddmm_cuda_general(sched, out)
        else:
            raise KeyError("Schedule type {} not recognized.".format(schedule_type))
    elif target == 'llvm':
        raise NotImplementedError('CPU kernel not implemented yet.')

    # prepare input
    f_input = out_feat_shp
    f_input.append(adj_row_indices)
    f_input.append(adj_col_indices)
    f_name = '_'.join(str(x) for x in [
        'sddmm', binary_op, ndim,
        indice_type, feat_type,
        lhs_target, rhs_target, schedule_type])
    f_input += [lhs, rhs, out]

    # bind autobroadcast buffer
    lhs_buffer = tvm.tir.decl_buffer(lhs.shape, lhs.dtype, name='lhs_buf',
                                     buffer_type='auto_broadcast')
    rhs_buffer = tvm.tir.decl_buffer(rhs.shape, rhs.dtype, name='rhs_buf',
                                     buffer_type='auto_broadcast')
    binds = {lhs:lhs_buffer, rhs:rhs_buffer}
    return tvm.lower(sched, f_input, name=f_name, binds=binds)

