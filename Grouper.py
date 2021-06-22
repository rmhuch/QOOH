import numpy as np

class Grouper:
    """
    A friendly little class for grouping data by some keys
    """
    
    def __init__(self, ar, keys):
        self.ar = ar
        if (
            isinstance(ar, np.ndarray) 
            and ar.ndim == 2
            and isinstance(keys, tuple)
            and len(keys) < ar.shape[1]
        ):
            keys = ar[:, keys]
        self.keys = keys
        self._groups = None
        self._inds = None
        self._sort = None
    
    @property
    def groups(self):
        if self._groups is None:
            self._groups, self._sort, self._inds = self.group_by(
                self.ar, self.keys, return_indices=True
            )
        return self._groups
        
    @property
    def group_keys(self):
        return self.groups[0]
    
    @property
    def group_vals(self):
        return self.groups[1]
    
    @property
    def group_dict(self):
        return {tuple(k): v for k, v in zip(*self.groups)}
    
    @classmethod
    def coerce_dtype(cls, ar, dtype=None):
        """
        Extracted from the way NumPy treats unique
        Coerces ar into a compound dtype so that it can be treated
        like a 1D array for set operations
        """

        # Must reshape to a contiguous 2D array for this to work...
        orig_shape, orig_dtype = ar.shape, ar.dtype
        ar = ar.reshape(orig_shape[0], np.prod(orig_shape[1:], dtype=np.intp))
        ar = np.ascontiguousarray(ar)
        if dtype is None:
            dtype = [('f{i}'.format(i=i), ar.dtype) for i in range(ar.shape[1])]
        # At this point, `ar` has shape `(n, m)`, and `dtype` is a structured
        # data type with `m` fields where each field has the data type of `ar`.
        # In the following, we create the array `consolidated`, which has
        # shape `(n,)` with data type `dtype`.
        try:
            if ar.shape[1] > 0:
                consolidated = ar.view(dtype)
                if len(consolidated.shape) > 1:
                    consolidated = consolidated.squeeze()
                    if consolidated.shape == ():
                        consolidated = np.expand_dims(consolidated, 0)
            else:
                # If ar.shape[1] == 0, then dtype will be `np.dtype([])`, which is
                # a data type with itemsize 0, and the call `ar.view(dtype)` will
                # fail.  Instead, we'll use `np.empty` to explicitly create the
                # array with shape `(len(ar),)`.  Because `dtype` in this case has
                # itemsize 0, the total size of the result is still 0 bytes.
                consolidated = np.empty(len(ar), dtype=dtype)
        except TypeError:
            # There's no good way to do this for object arrays, etc...
            msg = 'The axis argument to `coerce_dtype` is not supported for dtype {dt}'
            raise TypeError(msg.format(dt=ar.dtype))

        return consolidated, dtype, orig_shape, orig_dtype
    
    
    @classmethod
    def uncoerce_dtype(cls, consolidated, orig_shape, orig_dtype, axis):
        n = len(consolidated)
        uniq = consolidated.view(orig_dtype)
        uniq = uniq.reshape(n, *orig_shape[1:])
        if axis is not None:
            uniq = np.moveaxis(uniq, 0, axis)
        return uniq
    
    @classmethod
    def group_by1d(cls, ar, keys, return_indices=False):
        sorting = np.argsort(keys, kind='stable')
        uinds, mask = np.unique(keys[sorting], return_inverse=True)
        _, inds = np.unique(mask, return_index=True)
        groups = np.split(ar[sorting,], inds)[1:]
    
        ret = ((uinds, groups), sorting)
        if return_indices:
            ret += (inds,)
        return ret
    
    @classmethod
    def group_by(cls, ar, keys, return_indices=False):
        """
        Splits an array by a keys
        :param ar:
        :type ar:
        :param keys:
        :type keys:
        :param sorting:
        :type sorting:
        :return:
        :rtype:
        """

        ar = np.asanyarray(ar)
        keys = np.asanyarray(keys)

        if keys.ndim == 1:
            ret = cls.group_by1d(ar, keys, return_indices=return_indices)
            return ret

        keys, dtype, orig_shape, orig_dtype = cls.coerce_dtype(keys)
        output = cls.group_by1d(ar, keys, return_indices=return_indices)
        ukeys, groups = output[0]
        ukeys = cls.uncoerce_dtype(ukeys, orig_shape, orig_dtype, None)
        output = ((ukeys, groups),) + output[1:]
        return output