"""Columnar storage for DGLGraph."""
from __future__ import absolute_import

from collections import namedtuple
from collections.abc import MutableMapping

from . import backend as F
from .base import DGLError, dgl_warning
from .init import zero_initializer

class Scheme(namedtuple('Scheme', ['shape', 'dtype'])):
    """The column scheme.

    Parameters
    ----------
    shape : tuple of int
        The feature shape.
    dtype : backend-specific type object
        The feature data type.
    """
    # Pickling torch dtypes could be problemetic; this is a workaround.
    # I also have to create data_type_dict and reverse_data_type_dict
    # attribute just for this bug.
    # I raised an issue in PyTorch bug tracker:
    # https://github.com/pytorch/pytorch/issues/14057
    def __reduce__(self):
        state = (self.shape, F.reverse_data_type_dict[self.dtype])
        return self._reconstruct_scheme, state

    @classmethod
    def _reconstruct_scheme(cls, shape, dtype_str):
        dtype = F.data_type_dict[dtype_str]
        return cls(shape, dtype)

def infer_scheme(tensor):
    """Infer column scheme from the given tensor data.

    Parameters
    ---------
    tensor : Tensor
        The tensor data.

    Returns
    -------
    Scheme
        The column scheme.
    """
    return Scheme(tuple(F.shape(tensor)[1:]), F.dtype(tensor))

class Column(object):
    """A column is a compact store of features of multiple nodes/edges.

    It batches all the feature tensors together along the first dimension
    as one dense tensor.

    The column can optionally have an index tensor I.
    In this case, the i^th feature is stored in ``storage[index[i]]``.
    The column class implements a Copy-On-Read semantics -- the index
    select operation happens upon the first read of the feature data.
    This is useful when one extracts a subset of the feature data
    but wishes the actual index select happens on-demand.

    Parameters
    ----------
    storage : Tensor
        The feature data storage.
    scheme : Scheme, optional
        The scheme of the column. Will be inferred if not provided.
    index : Tensor, optional
        The row index to the feature data storage. None means an
        identity mapping.

    Attributes
    ----------
    storage : Tensor
        The storage tensor. The storage tensor may not be the actual data
        tensor of this column when the index tensor is not None.
        This typically happens when the column is extracted from another
        column using the `subcolumn` method.
    data : Tensor
        The actual data tensor of this column.
    scheme : Scheme
        The scheme of the column.
    index : Tensor
        Index tensor
    """
    def __init__(self, storage, scheme=None, index=None):
        self.storage = storage
        self.scheme = scheme if scheme else infer_scheme(storage)
        self.index = index

    def __len__(self):
        """The number of features (number of rows) in this column."""
        if self.index is None:
            return F.shape(self.storage)[0]
        else:
            return len(self.index)

    @property
    def shape(self):
        """Return the scheme shape (feature shape) of this column."""
        return self.scheme.shape

    @property
    def data(self):
        """Return the feature data. Perform index selecting if needed."""
        if self.index is not None:
            self.storage = F.gather_row(self.storage, self.index)
            self.index = None
        return self.storage

    @data.setter
    def data(self, val):
        """Update the column data."""
        self.index = None
        self.storage = val

    def __getitem__(self, rowids):
        """Return the feature data given the rowids.

        The operation triggers index selection.

        Parameters
        ----------
        rowids : Tensor
            Row ID tensor.

        Returns
        -------
        Tensor
            The feature data
        """
        return F.gather_row(self.data, rwoids)

    def __setitem__(self, rowids, feats):
        """Update the feature data given the index.

        The update is performed out-placely so it can be used in autograd mode.
        The operation triggers index selection.

        Parameters
        ----------
        rowids : Tensor
            Row IDs.
        feats : Tensor
            New features.
        """
        self.update(idx, feats)

    def update(self, rowids, feats):
        """Update the feature data given the index.

        Parameters
        ----------
        rowids : Tensor
            Row IDs.
        feats : Tensor
            New features.
        """
        feat_scheme = infer_scheme(feats)
        if feat_scheme != self.scheme:
            raise DGLError("Cannot update column of scheme %s using feature of scheme %s."
                           % (feat_scheme, self.scheme))
        self.data = F.scatter_row(self.data, rowids, feats)

    def extend(self, feats, feat_scheme=None):
        """Extend the feature data.

        The operation triggers index selection.

        Parameters
        ----------
        feats : Tensor
            The new features.
        feat_scheme : Scheme, optional
            The scheme
        """
        if feat_scheme is None:
            feat_scheme = infer_scheme(feats)

        if feat_scheme != self.scheme:
            raise DGLError("Cannot update column of scheme %s using feature of scheme %s."
                           % (feat_scheme, self.scheme))

        self.data = F.cat([self.data, feats], dim=0)

    def clone(self):
        """Return a shallow copy of this column."""
        return Column(self.storage, self.scheme, self.index)

    def deepclone(self):
        """Return a deepcopy of this column.

        The operation triggers index selection.
        """
        return Column(F.clone(self.data), self.scheme)

    def subcolumn(self, rowids):
        """Return a subcolumn.

        The resulting column will share the same storage as this column so this operation
        is quite efficient. If the current column is also a sub-column (i.e., the
        index tensor is not None), it slices the index tensor with the given
        rowids as the index tensor of the resulting column.

        Parameters
        ----------
        rowids : Tensor
            Row IDs.

        Returns
        -------
        Column
            Sub-column
        """
        if self.index is None:
            return Column(self.storage, self.scheme, rowids)
        else:
            return Column(self.storage, self.scheme, F.gather_row(self.index, rowids))

    @staticmethod
    def create(data):
        """Create a new column using the given data."""
        if isinstance(data, Column):
            return data.clone()
        else:
            return Column(data)

    def __repr__(self):
        return repr(self.data)

class Frame(MutableMapping):
    """The columnar storage for node/edge features.

    The frame is a dictionary from feature names to feature columns.
    All columns should have the same number of rows (i.e. the same first dimension).

    Parameters
    ----------
    data : dict-like, optional
        The frame data in dictionary. If the provided data is another frame,
        this frame will NOT share columns with the given frame. So any out-place
        update on one will not reflect to the other.
    num_rows : int, optional
        The number of rows in this frame. If ``data`` is provided and is not empty,
        ``num_rows`` will be ignored and inferred from the given data.
    """
    def __init__(self, data=None, num_rows=None):
        if data is None:
            self._columns = dict()
            self._num_rows = 0 if num_rows is None else num_rows
        else:
            assert not isinstance(data, Frame)  # sanity check for code refactor
            # Note that we always create a new column for the given data.
            # This avoids two frames accidentally sharing the same column.
            self._columns = {k : Column.create(v) for k, v in data.items()}
            self._num_rows = num_rows
            # infer num_rows & sanity check
            for name, col in self._columns.items():
                if self._num_rows is None:
                    self._num_rows = len(col)
                elif len(col) != self._num_rows:
                    raise DGLError('Expected all columns to have same # rows (%d), '
                                   'got %d on %r.' % (self._num_rows, len(col), name))

        # Initializer for empty values. Initializer is a callable.
        # If is none, then a warning will be raised
        # in the first call and zero initializer will be used later.
        self._initializers = {}  # per-column initializers
        self._default_initializer = None

    def _set_zero_default_initializer(self):
        """Set the default initializer to be zero initializer."""
        self._default_initializer = zero_initializer

    def get_initializer(self, column=None):
        """Get the initializer for empty values for the given column.

        Parameters
        ----------
        column : str
            The column

        Returns
        -------
        callable
            The initializer
        """
        return self._initializers.get(column, self._default_initializer)

    def set_initializer(self, initializer, column=None):
        """Set the initializer for empty values, for a given column or all future
        columns.

        Initializer is a callable that returns a tensor given the shape and data type.

        Parameters
        ----------
        initializer : callable
            The initializer.
        column : str, optional
            The column name
        """
        if column is None:
            self._default_initializer = initializer
        else:
            self._initializers[column] = initializer

    @property
    def schemes(self):
        """Return a dictionary of column name to column schemes."""
        return {k : col.scheme for k, col in self._columns.items()}

    @property
    def num_columns(self):
        """Return the number of columns in this frame."""
        return len(self._columns)

    @property
    def num_rows(self):
        """Return the number of rows in this frame."""
        return self._num_rows

    def __contains__(self, name):
        """Return true if the given column name exists."""
        return name in self._columns

    def __getitem__(self, name):
        """Return the column of the given name.

        Parameters
        ----------
        name : str
            The column name.

        Returns
        -------
        Tensor
            Column data.
        """
        return self._columns[name].data

    def __setitem__(self, name, data):
        """Update the whole column.

        Parameters
        ----------
        name : str
            The column name.
        col : Column or data convertible to Column
            The column data.
        """
        self.update_column(name, data)

    def __delitem__(self, name):
        """Delete the whole column.

        Parameters
        ----------
        name : str
            The column name.
        """
        del self._columns[name]

    def add_column(self, name, scheme, ctx):
        """Add a new column to the frame.

        The frame will be initialized by the initializer.

        Parameters
        ----------
        name : str
            The column name.
        scheme : Scheme
            The column scheme.
        ctx : DGLContext
            The column context.
        """
        if name in self:
            dgl_warning('Column "%s" already exists. Ignore adding this column again.' % name)
            return

        if self.get_initializer(name) is None:
            self._set_zero_default_initializer()
        initializer = self.get_initializer(name)
        init_data = initializer((self.num_rows,) + scheme.shape, scheme.dtype,
                                ctx, slice(0, self.num_rows))
        self._columns[name] = Column(init_data, scheme)

    def add_rows(self, num_rows):
        """Add blank rows to this frame.

        For existing fields, the rows will be extended according to their
        initializers.

        Parameters
        ----------
        num_rows : int
            The number of new rows
        """
        feat_placeholders = {}
        for key, col in self._columns.items():
            scheme = col.scheme
            ctx = F.context(col.data)
            if self.get_initializer(key) is None:
                self._set_zero_default_initializer()
            initializer = self.get_initializer(key)
            new_data = initializer((num_rows,) + scheme.shape, scheme.dtype,
                                   ctx, slice(self._num_rows, self._num_rows + num_rows))
            feat_placeholders[key] = new_data
        self._append(Frame(feat_placeholders))
        self._num_rows += num_rows

    def update_column(self, name, data):
        """Add or replace the column with the given name and data.

        Parameters
        ----------
        name : str
            The column name.
        data : Column or data convertible to Column
            The column data.
        """
        col = Column.create(data)
        if len(col) != self.num_rows:
            raise DGLError('Expected data to have %d rows, got %d.' %
                           (self.num_rows, len(col)))
        self._columns[name] = col

    def update_row(self, rowids, data):
        """Update the feature data of the given rows.

        If the data contains new keys (new columns) that do not exist in
        this frame, add a new column.

        The ``rowids`` shall not contain duplicates. Otherwise, the behavior
        is undefined.

        Parameters
        ----------
        rowids : Tensor
            Row Ids.
        data : dict[str, Tensor]
            Row data.
        """
        for key, val in data.items():
            if key not in self:
                scheme = infer_scheme(val)
                ctx = F.context(val)
                self.add_column(key, scheme, ctx)
        for key, val in data.items():
            self._columns[key].update(rowids, val)

    def _append(self, other):
        """Append ``other`` frame to ``self`` frame."""
        # NOTE: `other` can be empty.
        if self.num_rows == 0:
            # if no rows in current frame; append is equivalent to
            # directly updating columns.
            self._columns = {key: Column.create(data) for key, data in other.items()}
        else:
            # pad columns that are not provided in the other frame with initial values
            for key, col in self._columns.items():
                if key in other:
                    continue
                scheme = col.scheme
                ctx = F.context(col.data)
                if self.get_initializer(key) is None:
                    self._set_zero_default_initializer()
                initializer = self.get_initializer(key)
                new_data = initializer((other.num_rows,) + scheme.shape,
                                       scheme.dtype, ctx,
                                       slice(self._num_rows, self._num_rows + other.num_rows))
                other[key] = new_data
            # append other to self
            for key, col in other._columns.items():
                if key not in self._columns:
                    # the column does not exist; init a new column
                    self.add_column(key, col.scheme, F.context(col.data))
                self._columns[key].extend(col.data, col.scheme)

    def append(self, other):
        """Append another frame's data into this frame.

        If the current frame is empty, it will just use the columns of the
        given frame. Otherwise, the given data should contain all the
        column keys of this frame.

        Parameters
        ----------
        other : Frame or dict-like
            The frame data to be appended.
        """
        if not isinstance(other, Frame):
            other = Frame(other)
        self._append(other)
        self._num_rows += other.num_rows

    def clear(self):
        """Clear this frame. Remove all the columns."""
        self._columns = {}
        self._num_rows = 0

    def __iter__(self):
        """Return an iterator of columns."""
        return iter(self._columns)

    def __len__(self):
        """Return the number of columns."""
        return self.num_columns

    def keys(self):
        """Return the keys."""
        return self._columns.keys()

    def values(self):
        """Return the values."""
        return self._columns.values()

    def clone(self):
        """Return a clone of this frame.

        The clone frame does not share the underlying storage with this frame,
        i.e., adding or removing columns will not be visible to each other. However,
        they still share the tensor contents so any mutable operation on the column
        tensor are visible to each other. Hence, the function does not allocate extra
        tensor memory. Use :func:`~dgl.Frame.deepclone` for cloning
        a frame that does not share any data.

        Returns
        -------
        Frame
            A cloned frame.
        """
        newframe = Frame(self._columns, self._num_rows)
        newframe._initializers = self._initializers
        newframe._default_initializer = self._default_initializer
        return newframe

    def deepclone(self):
        """Return a deep clone of this frame.

        The clone frame has an copy of this frame and any modification to the clone frame
        is not visible to this frame. The function allocate new tensors and copy the contents
        from this frame. Use :func:`~dgl.Frame.clone` for cloning a frame that does not
        allocate extra tensor memory.

        Returns
        -------
        Frame
            A deep-cloned frame.
        """
        newframe = Frame({k : col.deepclone() for k, col in self._columns.items()},
                         self._num_rows)
        newframe._initializers = self._initializers
        newframe._default_initializer = self._default_initializer
        return newframe

    def subframe(self, rowids):
        """Return a new frame whose columns are subcolumns of this frame.

        The given row IDs should be within range [0, self.num_rows), and allow
        duplicate IDs.

        Parameters
        ----------
        rowids : Tensor
            Row IDs

        Returns
        -------
        Frame
            A new subframe.
        """
        subcols = {k : col.subcolumn(rowids) for k, col in self._columns.items()}
        subf = Frame(subcols, len(rowids))
        subf._initializers = self._initializers
        subf._default_initializer = self._default_initializer
        return subf

    def __repr__(self):
        return repr(dict(self))
