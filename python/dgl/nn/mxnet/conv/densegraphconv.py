"""MXNet Module for DenseGraphConv"""
# pylint: disable= no-member, arguments-differ, invalid-name
import mxnet as mx
import math
from mxnet import nd
from mxnet.gluon import nn


class DenseGraphConv(nn.Block):
    """Graph Convolutional Network layer where the graph structure
    is given by an adjacency matrix.
    We recommend user to use this module when inducing graph convolution
    on dense graphs / k-hop graphs.

    Parameters
    ----------
    in_feats : int
        Input feature size.
    out_feats : int
        Output feature size.
    norm : bool
        If True, the normalizer :math:`c_{ij}` is applied. Default: ``True``.
    bias : bool
        If True, adds a learnable bias to the output. Default: ``True``.
    activation : callable activation function/layer or None, optional
        If not None, applies an activation function to the updated node features.
        Default: ``None``.

    See also
    --------
    GraphConv
    """
    def __init__(self,
                 in_feats,
                 out_feats,
                 norm=True,
                 bias=True,
                 activation=None):
        super(DenseGraphConv, self).__init__()
        self._in_feats = in_feats
        self._out_feats = out_feats
        self._norm = norm
        with self.name_scope():
            self.weight = self.params.get('weight', shape=(in_feats, out_feats),
                                          init=mx.init.Xavier(magnitude=math.sqrt(2.0)))
            if bias:
                self.bias = self.params.get('bias', shape=(out_feats,),
                                            init=mx.init.Zero())
            else:
                self.bias = None
            self._activation = activation

    def forward(self, adj, feat):
        r"""Compute (Dense) Graph Convolution layer.

        Parameters
        ----------
        adj : mxnet.NDArray
            The adjacency matrix of the graph to apply Graph Convolution on,
            should be of shape :math:`(N, N)`, where a row represents the destination
            and a column represents the source.
        feat : mxnet.NDArray
            The input feature of shape :math:`(N, D_{in})` where :math:`D_{in}`
            is size of input feature, :math:`N` is the number of nodes.

        Returns
        -------
        mxnet.NDArray
            The output feature of shape :math:`(N, D_{out})` where :math:`D_{out}`
            is size of output feature.
        """
        pass
